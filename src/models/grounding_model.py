import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_fusion import CrossModalFusion
from src.data.grounding_dataset import GroundingDataset


class GroundingModel(nn.Module):
    """
    Spatial-semantic grounding model.

    Inputs:
        s1              [B, 196, 768]
        s2              [B, 196, 768]
        text_embedding  [B, 768]
        point           [B, 2]
        mode_id         [B]   0=reference, 1=point

    Output:
        boxes           [B, 4]
                        [x1, y1, x2, y2]
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ----------------------------------------------------
        # Visual fusion
        # ----------------------------------------------------

        self.visual_fusion = CrossModalFusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        # ----------------------------------------------------
        # Query projection
        # ----------------------------------------------------

        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Point projection
        # ----------------------------------------------------

        self.point_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Mode embedding
        # ----------------------------------------------------

        self.mode_embedding = nn.Embedding(
            2,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Spatial positional embedding
        # ----------------------------------------------------

        self.spatial_pos = nn.Parameter(
            torch.zeros(
                1,
                196,
                hidden_dim,
            )
        )

        nn.init.normal_(
            self.spatial_pos,
            std=0.02,
        )

        # ----------------------------------------------------
        # Per-token semantic matching
        #
        # Each image token is compared explicitly with the
        # query rather than reducing the image to one vector.
        # ----------------------------------------------------

        self.visual_match = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.query_match = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.match_score = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        # ----------------------------------------------------
        # Point-guidance temperature
        # ----------------------------------------------------

        self.point_temperature = nn.Parameter(
            torch.tensor(10.0)
        )

        # ----------------------------------------------------
        # Spatial refinement
        # ----------------------------------------------------

        self.spatial_refine = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Box head
        # ----------------------------------------------------

        self.box_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 5,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                4,
            ),
        )

    def build_point_attention(
        self,
        point: torch.Tensor,
    ) -> torch.Tensor:
        """
        point: [B,2], normalized [0,1]
        returns: [B,196]
        """

        device = point.device

        grid_coords = torch.linspace(
            0.5 / 14.0,
            1.0 - 0.5 / 14.0,
            14,
            device=device,
        )

        yy, xx = torch.meshgrid(
            grid_coords,
            grid_coords,
            indexing="ij",
        )

        grid = torch.stack(
            [
                xx.reshape(-1),
                yy.reshape(-1),
            ],
            dim=-1,
        )
        # [196,2]

        distance_sq = (
            (
                point.unsqueeze(1)
                - grid.unsqueeze(0)
            ) ** 2
        ).sum(dim=-1)

        temperature = (
            F.softplus(
                self.point_temperature
            )
            + 1e-4
        )

        weights = torch.softmax(
            -distance_sq * temperature,
            dim=1,
        )

        return weights

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        text_embedding: torch.Tensor,
        point: torch.Tensor,
        mode_id: torch.Tensor,
    ):

        # ----------------------------------------------------
        # 1. Fuse modalities
        # ----------------------------------------------------

        visual = self.visual_fusion(
            s1,
            s2,
        )
        # [B,196,256]

        visual = (
            visual
            + self.spatial_pos
        )

        # ----------------------------------------------------
        # 2. Query / point / mode
        # ----------------------------------------------------

        query = self.query_proj(
            text_embedding
        )
        # [B,256]

        point_repr = self.point_proj(
            point
        )
        # [B,256]

        mode_repr = self.mode_embedding(
            mode_id.long()
        )
        # [B,256]

        # ----------------------------------------------------
        # 3. Per-token semantic matching
        # ----------------------------------------------------

        visual_match = self.visual_match(
            visual
        )
        # [B,196,256]

        query_match = self.query_match(
            query
        ).unsqueeze(1)
        # [B,1,256]

        match_features = torch.tanh(
            visual_match
            + query_match
        )

        semantic_scores = self.match_score(
            match_features
        ).squeeze(-1)
        # [B,196]

        semantic_weights = torch.softmax(
            semantic_scores,
            dim=1,
        )

        # ----------------------------------------------------
        # 4. Point-guided attention
        # ----------------------------------------------------

        point_weights = self.build_point_attention(
            point
        )

        is_point = (
            mode_id == 1
        ).float().unsqueeze(1)

        # Point mode:
        # blend explicit spatial guidance with semantic matching.
        #
        # Reference mode:
        # rely entirely on semantic matching.

        spatial_weights = (
            (1.0 - is_point)
            * semantic_weights
            + is_point
            * (
                0.75 * point_weights
                + 0.25 * semantic_weights
            )
        )

        spatial_weights = (
            spatial_weights
            / spatial_weights.sum(
                dim=1,
                keepdim=True,
            ).clamp(min=1e-8)
        )

        # ----------------------------------------------------
        # 5. Query-conditioned spatial representation
        # ----------------------------------------------------

        spatial_summary = torch.sum(
            visual
            * spatial_weights.unsqueeze(-1),
            dim=1,
        )

        spatial_summary = self.spatial_refine(
            spatial_summary
        )
        # [B,256]

        # ----------------------------------------------------
        # 6. Global visual context
        # ----------------------------------------------------

        global_visual = visual.mean(
            dim=1
        )
        # [B,256]

        # ----------------------------------------------------
        # 7. Semantic similarity statistics
        # ----------------------------------------------------

        max_semantic_score = (
            semantic_scores.max(
                dim=1
            ).values
        )

        mean_semantic_score = (
            semantic_scores.mean(
                dim=1
            )
        )

        # These are [B], so convert to [B,1].
        max_semantic_score = (
            max_semantic_score.unsqueeze(1)
        )

        mean_semantic_score = (
            mean_semantic_score.unsqueeze(1)
        )

        # ----------------------------------------------------
        # 8. Final representation
        #
        # spatial summary     256
        # global visual       256
        # query               256
        # point               256
        # mode                256
        # -------------------------
        # total              1280
        # ----------------------------------------------------

        representation = torch.cat(
            [
                spatial_summary,
                global_visual,
                query,
                point_repr,
                mode_repr,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # 9. Predict cx,cy,w,h
        # ----------------------------------------------------

        raw_box = self.box_head(
            representation
        )

        cxcywh = torch.sigmoid(
            raw_box
        )

        cx = cxcywh[:, 0]
        cy = cxcywh[:, 1]
        w = cxcywh[:, 2]
        h = cxcywh[:, 3]

        x1 = (
            cx - w / 2
        ).clamp(0.0, 1.0)

        y1 = (
            cy - h / 2
        ).clamp(0.0, 1.0)

        x2 = (
            cx + w / 2
        ).clamp(0.0, 1.0)

        y2 = (
            cy + h / 2
        ).clamp(0.0, 1.0)

        return torch.stack(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dim=1,
        )


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:

    x1 = torch.maximum(
        boxes1[:, 0],
        boxes2[:, 0],
    )

    y1 = torch.maximum(
        boxes1[:, 1],
        boxes2[:, 1],
    )

    x2 = torch.minimum(
        boxes1[:, 2],
        boxes2[:, 2],
    )

    y2 = torch.minimum(
        boxes1[:, 3],
        boxes2[:, 3],
    )

    intersection = (
        (x2 - x1).clamp(min=0.0)
        *
        (y2 - y1).clamp(min=0.0)
    )

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0)
        *
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0)
        *
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
    )

    union = (
        area1
        + area2
        - intersection
    )

    return (
        intersection
        / union.clamp(min=1e-8)
    )


if __name__ == "__main__":

    print("=" * 70)
    print("SEMANTIC-SPATIAL GROUNDING MODEL SMOKE TEST")
    print("=" * 70)

    dataset = GroundingDataset()

    indices = [
        0,
        1,
        2,
        3,
        4,
        1000,
        10000,
        20000,
    ]

    samples = [
        dataset[i]
        for i in indices
    ]

    s1 = torch.stack(
        [x["s1"] for x in samples]
    )

    s2 = torch.stack(
        [x["s2"] for x in samples]
    )

    text_embedding = torch.stack(
        [x["text_embedding"] for x in samples]
    )

    point = torch.stack(
        [x["point"] for x in samples]
    )

    target_boxes = torch.stack(
        [x["target_box"] for x in samples]
    )

    mode_id = torch.stack(
        [x["mode_id"] for x in samples]
    )

    print(
        f"S1:              {tuple(s1.shape)}"
    )

    print(
        f"S2:              {tuple(s2.shape)}"
    )

    print(
        f"Text embedding:  "
        f"{tuple(text_embedding.shape)}"
    )

    print(
        f"Point:           {tuple(point.shape)}"
    )

    print(
        f"Target box:      {tuple(target_boxes.shape)}"
    )

    print(
        f"Mode ID:         {tuple(mode_id.shape)}"
    )

    model = GroundingModel()

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    predicted_boxes = model(
        s1=s1,
        s2=s2,
        text_embedding=text_embedding,
        point=point,
        mode_id=mode_id,
    )

    print(
        f"Predicted boxes: "
        f"{tuple(predicted_boxes.shape)}"
    )

    print("\nPredictions:")

    for i in range(
        len(samples)
    ):
        print(
            f"Example {i + 1}: "
            f"mode={samples[i]['mode']} "
            f"target={target_boxes[i].tolist()} "
            f"pred={predicted_boxes[i].tolist()}"
        )

    loss = F.smooth_l1_loss(
        predicted_boxes,
        target_boxes,
    )

    ious = box_iou(
        predicted_boxes,
        target_boxes,
    )

    print(
        f"\nInitial SmoothL1 loss: "
        f"{loss.item():.6f}"
    )

    print(
        f"Initial mean IoU: "
        f"{ious.mean().item():.6f}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    optimizer.zero_grad()

    loss.backward()

    nonzero_grad_params = sum(
        1
        for p in model.parameters()
        if p.grad is not None
        and p.grad.abs().sum().item() > 0
    )

    optimizer.step()

    print(
        f"Parameters with nonzero gradients: "
        f"{nonzero_grad_params}"
    )

    print("\n" + "=" * 70)
    print(
        "SEMANTIC-SPATIAL GROUNDING SMOKE TEST PASSED"
    )
    print("=" * 70)