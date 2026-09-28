import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Bounding-box IoU
# ============================================================

def box_iou(boxes1, boxes2, eps=1e-6):
    """
    Calculate IoU between two batches of boxes.

    Format:
        [x1, y1, x2, y2]
    """

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

    inter_w = (
        x2 - x1
    ).clamp(min=0.0)

    inter_h = (
        y2 - y1
    ).clamp(min=0.0)

    intersection = (
        inter_w * inter_h
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

    return intersection / (
        union + eps
    )


# ============================================================
# Boxes -> target heatmap
# ============================================================

def boxes_to_heatmaps(
    boxes,
    grid_size=14,
):
    """
    Convert normalized boxes into soft 14x14 spatial maps.

    Each cell receives the fraction of its area covered by the
    ground-truth box.
    """

    device = boxes.device
    dtype = boxes.dtype

    batch_size = boxes.size(0)

    edges = torch.linspace(
        0.0,
        1.0,
        grid_size + 1,
        device=device,
        dtype=dtype,
    )

    cell_x1 = edges[:-1].view(
        1,
        1,
        grid_size,
    )

    cell_x2 = edges[1:].view(
        1,
        1,
        grid_size,
    )

    cell_y1 = edges[:-1].view(
        1,
        grid_size,
        1,
    )

    cell_y2 = edges[1:].view(
        1,
        grid_size,
        1,
    )

    bx1 = boxes[:, 0].view(
        batch_size,
        1,
        1,
    )

    by1 = boxes[:, 1].view(
        batch_size,
        1,
        1,
    )

    bx2 = boxes[:, 2].view(
        batch_size,
        1,
        1,
    )

    by2 = boxes[:, 3].view(
        batch_size,
        1,
        1,
    )

    inter_x1 = torch.maximum(
        cell_x1,
        bx1,
    )

    inter_x2 = torch.minimum(
        cell_x2,
        bx2,
    )

    inter_y1 = torch.maximum(
        cell_y1,
        by1,
    )

    inter_y2 = torch.minimum(
        cell_y2,
        by2,
    )

    inter_w = (
        inter_x2 - inter_x1
    ).clamp(min=0.0)

    inter_h = (
        inter_y2 - inter_y1
    ).clamp(min=0.0)

    intersection = (
        inter_w * inter_h
    )

    cell_area = 1.0 / (
        grid_size * grid_size
    )

    heatmap = (
        intersection / cell_area
    ).clamp(
        min=0.0,
        max=1.0,
    )

    return heatmap


# ============================================================
# Spatial softmax loss
# ============================================================

def spatial_softmax_loss(
    logits,
    target_heatmap,
    temperature=1.0,
    eps=1e-8,
):
    """
    Treat all 196 spatial cells as one probability distribution.
    """

    batch_size = logits.size(0)

    logits_flat = logits.reshape(
        batch_size,
        -1,
    )

    target_flat = target_heatmap.reshape(
        batch_size,
        -1,
    )

    target_distribution = (
        target_flat
        /
        (
            target_flat.sum(
                dim=1,
                keepdim=True,
            )
            + eps
        )
    )

    log_pred_distribution = F.log_softmax(
        logits_flat / temperature,
        dim=1,
    )

    loss = -(
        target_distribution
        * log_pred_distribution
    ).sum(dim=1)

    return loss.mean()


# ============================================================
# Spatial geometry extraction
# ============================================================

def extract_spatial_geometry(
    heatmap,
    grid_size=14,
    eps=1e-8,
):
    """
    Extract differentiable spatial geometry from the predicted
    probability heatmap.

    Returns 8 values:

        0 = center_x
        1 = center_y
        2 = std_x
        3 = std_y
        4 = estimated_x1
        5 = estimated_y1
        6 = estimated_x2
        7 = estimated_y2
    """

    batch_size = heatmap.size(0)

    device = heatmap.device
    dtype = heatmap.dtype

    coords = (
        torch.arange(
            grid_size,
            device=device,
            dtype=dtype,
        )
        + 0.5
    ) / grid_size

    x_coords = coords.view(
        1,
        1,
        grid_size,
    )

    y_coords = coords.view(
        1,
        grid_size,
        1,
    )

    # --------------------------------------------------------
    # Spatial center
    # --------------------------------------------------------

    center_x = (
        heatmap
        * x_coords
    ).sum(dim=(1, 2))

    center_y = (
        heatmap
        * y_coords
    ).sum(dim=(1, 2))

    # --------------------------------------------------------
    # Spatial variance
    # --------------------------------------------------------

    center_x_expanded = center_x.view(
        batch_size,
        1,
        1,
    )

    center_y_expanded = center_y.view(
        batch_size,
        1,
        1,
    )

    variance_x = (
        heatmap
        * (
            x_coords
            - center_x_expanded
        ).pow(2)
    ).sum(dim=(1, 2))

    variance_y = (
        heatmap
        * (
            y_coords
            - center_y_expanded
        ).pow(2)
    ).sum(dim=(1, 2))

    std_x = torch.sqrt(
        variance_x + eps
    )

    std_y = torch.sqrt(
        variance_y + eps
    )

    # --------------------------------------------------------
    # Approximate rectangle dimensions
    #
    # For a uniform rectangle:
    #
    #     std = width / sqrt(12)
    #
    # Therefore:
    #
    #     width ≈ std * sqrt(12)
    # --------------------------------------------------------

    estimated_width = (
        std_x
        * (12.0 ** 0.5)
    )

    estimated_height = (
        std_y
        * (12.0 ** 0.5)
    )

    estimated_width = estimated_width.clamp(
        min=0.02,
        max=1.0,
    )

    estimated_height = estimated_height.clamp(
        min=0.02,
        max=1.0,
    )

    # --------------------------------------------------------
    # Base geometry box
    # --------------------------------------------------------

    estimated_x1 = (
        center_x
        - estimated_width / 2.0
    ).clamp(
        min=0.0,
        max=1.0,
    )

    estimated_y1 = (
        center_y
        - estimated_height / 2.0
    ).clamp(
        min=0.0,
        max=1.0,
    )

    estimated_x2 = (
        center_x
        + estimated_width / 2.0
    ).clamp(
        min=0.0,
        max=1.0,
    )

    estimated_y2 = (
        center_y
        + estimated_height / 2.0
    ).clamp(
        min=0.0,
        max=1.0,
    )

    geometry = torch.stack(
        [
            center_x,
            center_y,
            std_x,
            std_y,
            estimated_x1,
            estimated_y1,
            estimated_x2,
            estimated_y2,
        ],
        dim=-1,
    )

    return geometry


# ============================================================
# Grounding Heatmap Model
# ============================================================

class GroundingHeatmapModel(nn.Module):
    """
    Reference grounding model.

    Stage 1:
        S1 + S2 + text
        -> 196 spatial compatibility scores
        -> 14x14 probability heatmap

    Stage 2:
        heatmap
        -> spatial geometry
        -> geometry-derived base box
        -> learned box correction
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        grid_size=14,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.hidden_dim = hidden_dim

        # ----------------------------------------------------
        # Visual fusion
        # ----------------------------------------------------

        self.visual_fusion = nn.Sequential(
            nn.Linear(
                input_dim * 2,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim
            ),
        )

        # ----------------------------------------------------
        # Query projection
        # ----------------------------------------------------

        self.query_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim
            ),
        )

        # ----------------------------------------------------
        # Spatial positions
        # ----------------------------------------------------

        self.spatial_pos = nn.Parameter(
            torch.randn(
                1,
                grid_size * grid_size,
                hidden_dim,
            )
            * 0.02
        )

        # ----------------------------------------------------
        # Visual semantic projection
        # ----------------------------------------------------

        self.visual_match = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Query semantic projection
        # ----------------------------------------------------

        self.query_match = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.match_norm = nn.LayerNorm(
            hidden_dim
        )

        # ----------------------------------------------------
        # Localization score
        # ----------------------------------------------------

        self.match_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

        # ----------------------------------------------------
        # Global visual feature
        # ----------------------------------------------------

        self.spatial_summary = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim
            ),
        )

        # ----------------------------------------------------
        # Box correction input
        #
        # local feature: 256
        # global feature: 256
        # query: 256
        # geometry: 8
        # base box: 4
        #
        # total = 780
        # ----------------------------------------------------

        refinement_dim = (
            hidden_dim * 3
            + 8
            + 4
        )

        self.box_correction_head = nn.Sequential(
            nn.Linear(
                refinement_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim // 2,
                4,
            ),
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        s1,
        s2,
        text_embedding,
    ):
        batch_size = s1.size(0)

        # ----------------------------------------------------
        # 1. Fuse S1 + S2
        # ----------------------------------------------------

        visual = torch.cat(
            [
                s1,
                s2,
            ],
            dim=-1,
        )

        visual = self.visual_fusion(
            visual
        )

        # [B,196,256]

        # ----------------------------------------------------
        # 2. Spatial position
        # ----------------------------------------------------

        visual = (
            visual
            + self.spatial_pos
        )

        # ----------------------------------------------------
        # 3. Query
        # ----------------------------------------------------

        query = self.query_proj(
            text_embedding
        )

        # [B,256]

        # ----------------------------------------------------
        # 4. Cross-modal compatibility
        # ----------------------------------------------------

        visual_match = self.visual_match(
            visual
        )

        query_match = self.query_match(
            query
        ).unsqueeze(1)

        joint = (
            visual_match
            + query_match
        )

        joint = self.match_norm(
            F.gelu(joint)
        )

        # ----------------------------------------------------
        # 5. Localization logits
        # ----------------------------------------------------

        heatmap_logits = self.match_head(
            joint
        ).squeeze(-1)

        heatmap_logits = heatmap_logits.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
        )

        # ----------------------------------------------------
        # 6. Spatial probability
        # ----------------------------------------------------

        weights = torch.softmax(
            heatmap_logits.reshape(
                batch_size,
                self.grid_size
                * self.grid_size,
            ),
            dim=1,
        )

        weights_2d = weights.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
        )

        # ----------------------------------------------------
        # 7. Heatmap-selected local feature
        # ----------------------------------------------------

        local_feature = (
            visual
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        # ----------------------------------------------------
        # 8. Global visual feature
        # ----------------------------------------------------

        global_visual = visual.mean(
            dim=1
        )

        global_visual = self.spatial_summary(
            global_visual
        )

        # ----------------------------------------------------
        # 9. Extract explicit spatial geometry
        # ----------------------------------------------------

        geometry = extract_spatial_geometry(
            weights_2d,
            grid_size=self.grid_size,
        )

        # ----------------------------------------------------
        # 10. Base box comes DIRECTLY from heatmap geometry
        # ----------------------------------------------------

        base_box = geometry[
            :,
            4:8,
        ]

        # ----------------------------------------------------
        # 11. Prepare correction features
        # ----------------------------------------------------

        correction_input = torch.cat(
            [
                local_feature,
                global_visual,
                query,
                geometry,
                base_box,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # 12. Predict SMALL coordinate corrections
        # ----------------------------------------------------

        correction_raw = (
            self.box_correction_head(
                correction_input
            )
        )

        # Limit correction magnitude.
        #
        # tanh -> [-1,1]
        # * 0.20 -> [-0.20,0.20]
        #
        # The model therefore modifies the geometry estimate
        # rather than completely replacing it.
        # ----------------------------------------------------

        correction = (
            torch.tanh(
                correction_raw
            )
            * 0.20
        )

        # ----------------------------------------------------
        # 13. Correct the geometry-derived box
        # ----------------------------------------------------

        box = (
            base_box
            + correction
        )

        # ----------------------------------------------------
        # 14. Keep coordinates valid
        # ----------------------------------------------------

        x1 = box[:, 0].clamp(
            min=0.0,
            max=1.0,
        )

        y1 = box[:, 1].clamp(
            min=0.0,
            max=1.0,
        )

        x2 = box[:, 2].clamp(
            min=0.0,
            max=1.0,
        )

        y2 = box[:, 3].clamp(
            min=0.0,
            max=1.0,
        )

        # ----------------------------------------------------
        # Ensure valid ordering
        # ----------------------------------------------------

        final_x1 = torch.minimum(
            x1,
            x2,
        )

        final_y1 = torch.minimum(
            y1,
            y2,
        )

        final_x2 = torch.maximum(
            x1,
            x2,
        )

        final_y2 = torch.maximum(
            y1,
            y2,
        )

        final_box = torch.stack(
            [
                final_x1,
                final_y1,
                final_x2,
                final_y2,
            ],
            dim=-1,
        )

        return {
            "box": final_box,
            "base_box": base_box,
            "correction": correction,
            "geometry": geometry,
            "heatmap_logits": heatmap_logits,
            "heatmap": weights_2d,
        }


# ============================================================
# Combined loss
# ============================================================

def grounding_heatmap_loss(
    outputs,
    target_boxes,
    heatmap_weight=1.0,
    box_weight=0.25,
):
    """
    Heatmap localization + final box correction loss.
    """

    target_heatmap = boxes_to_heatmaps(
        target_boxes,
        grid_size=14,
    )

    heatmap_loss = spatial_softmax_loss(
        outputs["heatmap_logits"],
        target_heatmap,
    )

    box_loss = F.smooth_l1_loss(
        outputs["box"],
        target_boxes,
    )

    total_loss = (
        heatmap_weight
        * heatmap_loss
        +
        box_weight
        * box_loss
    )

    return {
        "total": total_loss,
        "heatmap": heatmap_loss,
        "box": box_loss,
        "target_heatmap": target_heatmap,
    }