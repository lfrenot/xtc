from collections.abc import Sequence
from .node import XTCNode


__all__ = [
    "XTCGraphUtils",
]


class XTCGraphUtils:
    @staticmethod
    def get_nodes_outputs(nodes: Sequence[XTCNode]) -> Sequence[XTCNode]:
        unique_nodes = list({node.uid: node for node in nodes}.values())
        assert len(unique_nodes) == len(nodes)
        used = set()
        for node in nodes:
            for pred_node in node.preds_nodes:
                used.add(pred_node)
        outputs = [node for node in nodes if node not in used]
        return outputs

    @staticmethod
    def get_nodes_topological_from_seed(nodes: Sequence[XTCNode], seed: Sequence[XTCNode]) -> Sequence[XTCNode]:
        unique_nodes = list({node.uid: node for node in nodes}.values())
        assert len(unique_nodes) == len(nodes)
        nodes_set = set(nodes)
        assert all([node in nodes_set for node in seed])
        seen = set()
        rwalk = []
        def reverse_walk(out: XTCNode) -> None:
            if out in seen or out not in nodes_set:
                return
            seen.add(out)
            for pred_node in out.preds_nodes:
                reverse_walk(pred_node)
            rwalk.append(out)
        for node in seed:
            reverse_walk(node)
        return rwalk

    @staticmethod
    def get_nodes_topological(nodes: Sequence[XTCNode]) -> Sequence[XTCNode]:
        outputs = XTCGraphUtils.get_nodes_outputs(nodes)
        sorted = XTCGraphUtils.get_nodes_topological_from_seed(nodes, outputs)
        return sorted
