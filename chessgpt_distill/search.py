"""Search algorithms for ChessGPT: greedy, MCTS with value head, MCTS with SF oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import torch
import torch.nn.functional as F

from chessgpt import ChessGPT, UCITokenizer

from .stockfish import StockfishPool


# ---------------------------------------------------------------------------
# Inference primitive
# ---------------------------------------------------------------------------


def _encode_history(
    tokenizer: UCITokenizer,
    move_history: list[str],
    max_seq_len: int,
) -> list[int]:
    ids = [tokenizer.BOS_ID]
    for mv in move_history:
        tid = tokenizer.move_to_id.get(mv)
        if tid is not None:
            ids.append(tid)
    if len(ids) > max_seq_len:
        ids = ids[-max_seq_len:]
    return ids


def _build_legal_mask(
    board: chess.Board,
    tokenizer: UCITokenizer,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
    for mv in board.legal_moves:
        tid = tokenizer.move_to_id.get(mv.uci())
        if tid is not None:
            mask[tid] = True
    return mask


@torch.no_grad()
def get_policy_and_value(
    model: ChessGPT,
    tokenizer: UCITokenizer,
    move_history: list[str],
    board: chess.Board,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    """Single forward pass returning policy over legal moves and position value.

    Returns:
        policy: dict mapping UCI move string -> probability (legal moves only)
        value: float in [-1, +1] from white's perspective
    """
    ids = _encode_history(tokenizer, move_history, model.config.max_seq_len)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits, values, _ = model(input_ids)

    # Policy from last position
    move_logits = logits[0, -1, :]  # (V,)
    move_logits[tokenizer.PAD_ID] = float("-inf")
    move_logits[tokenizer.BOS_ID] = float("-inf")
    move_logits[tokenizer.EOS_ID] = float("-inf")

    legal_mask = _build_legal_mask(board, tokenizer, device)
    move_logits = move_logits.masked_fill(~legal_mask, float("-inf"))

    probs = F.softmax(move_logits, dim=-1)

    policy = {}
    for mv in board.legal_moves:
        uci = mv.uci()
        tid = tokenizer.move_to_id.get(uci)
        if tid is not None:
            policy[uci] = probs[tid].item()

    value = values[0, -1].item()
    return policy, value


# ---------------------------------------------------------------------------
# Greedy search (with legal masking)
# ---------------------------------------------------------------------------


def search_greedy_legal(
    model: ChessGPT,
    tokenizer: UCITokenizer,
    move_history: list[str],
    board: chess.Board,
    device: torch.device,
) -> str | None:
    """Greedy argmax over legal moves."""
    policy, _ = get_policy_and_value(model, tokenizer, move_history, board, device)
    if not policy:
        return None
    return max(policy, key=policy.get)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    move: str | None = None
    parent: MCTSNode | None = None
    children: dict[str, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _select_child(node: MCTSNode, c_puct: float) -> str:
    """Select child via PUCT formula."""
    sqrt_parent = math.sqrt(node.visit_count)
    best_score = float("-inf")
    best_move = None
    for move, child in node.children.items():
        exploitation = child.q_value
        exploration = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        score = exploitation + exploration
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def _backup(node: MCTSNode, value: float) -> None:
    """Propagate value up the tree, alternating sign at each level."""
    v = value
    while node is not None:
        node.visit_count += 1
        node.value_sum += v
        v = -v
        node = node.parent


def mcts_search(
    model: ChessGPT,
    tokenizer: UCITokenizer,
    move_history: list[str],
    board: chess.Board,
    device: torch.device,
    num_simulations: int = 100,
    c_puct: float = 1.5,
) -> str | None:
    """MCTS using the model's policy and value heads.

    Returns the best move UCI string, or None if no legal moves.
    """
    if board.is_game_over() or not list(board.legal_moves):
        return None

    root = MCTSNode()

    # Expand root
    root_policy, root_value = get_policy_and_value(
        model, tokenizer, move_history, board, device,
    )
    for move_uci, prior in root_policy.items():
        root.children[move_uci] = MCTSNode(
            move=move_uci, parent=root, prior=prior,
        )
    # Backup root value (from side-to-move perspective)
    v_stm = root_value if board.turn == chess.WHITE else -root_value
    _backup(root, v_stm)

    for _ in range(num_simulations):
        node = root
        sim_board = board.copy()
        sim_history = list(move_history)

        # Select
        while node.children and node.visit_count > 0:
            move_uci = _select_child(node, c_puct)
            node = node.children[move_uci]
            sim_board.push_uci(move_uci)
            sim_history.append(move_uci)

        # Terminal check
        if sim_board.is_game_over():
            result = sim_board.result()
            if result == "1-0":
                leaf_value = 1.0
            elif result == "0-1":
                leaf_value = -1.0
            else:
                leaf_value = 0.0
            # Convert to side-to-move perspective at leaf
            v_stm = leaf_value if sim_board.turn == chess.WHITE else -leaf_value
            _backup(node, v_stm)
            continue

        # Expand
        policy, value = get_policy_and_value(
            model, tokenizer, sim_history, sim_board, device,
        )
        for move_uci, prior in policy.items():
            if move_uci not in node.children:
                node.children[move_uci] = MCTSNode(
                    move=move_uci, parent=node, prior=prior,
                )

        # Backup (value is from white's perspective, convert to side-to-move)
        v_stm = value if sim_board.turn == chess.WHITE else -value
        _backup(node, v_stm)

    # Select most-visited child
    if not root.children:
        return None
    return max(root.children, key=lambda m: root.children[m].visit_count)


def mcts_search_sf(
    model: ChessGPT,
    tokenizer: UCITokenizer,
    move_history: list[str],
    board: chess.Board,
    device: torch.device,
    pool: StockfishPool,
    sf_depth: int = 3,
    num_simulations: int = 100,
    c_puct: float = 1.5,
) -> str | None:
    """MCTS using model policy but Stockfish as value oracle (benchmark)."""
    if board.is_game_over() or not list(board.legal_moves):
        return None

    root = MCTSNode()

    # Expand root
    root_policy, _ = get_policy_and_value(
        model, tokenizer, move_history, board, device,
    )
    cp = pool.evaluate(board, depth=sf_depth)
    root_value = math.tanh(cp / 400.0)

    for move_uci, prior in root_policy.items():
        root.children[move_uci] = MCTSNode(
            move=move_uci, parent=root, prior=prior,
        )
    v_stm = root_value if board.turn == chess.WHITE else -root_value
    _backup(root, v_stm)

    for _ in range(num_simulations):
        node = root
        sim_board = board.copy()
        sim_history = list(move_history)

        # Select
        while node.children and node.visit_count > 0:
            move_uci = _select_child(node, c_puct)
            node = node.children[move_uci]
            sim_board.push_uci(move_uci)
            sim_history.append(move_uci)

        # Terminal
        if sim_board.is_game_over():
            result = sim_board.result()
            if result == "1-0":
                leaf_value = 1.0
            elif result == "0-1":
                leaf_value = -1.0
            else:
                leaf_value = 0.0
            v_stm = leaf_value if sim_board.turn == chess.WHITE else -leaf_value
            _backup(node, v_stm)
            continue

        # Expand with model policy, evaluate with Stockfish
        policy, _ = get_policy_and_value(
            model, tokenizer, sim_history, sim_board, device,
        )
        for move_uci, prior in policy.items():
            if move_uci not in node.children:
                node.children[move_uci] = MCTSNode(
                    move=move_uci, parent=node, prior=prior,
                )

        cp = pool.evaluate(sim_board, depth=sf_depth)
        value = math.tanh(cp / 400.0)
        v_stm = value if sim_board.turn == chess.WHITE else -value
        _backup(node, v_stm)

    if not root.children:
        return None
    return max(root.children, key=lambda m: root.children[m].visit_count)
