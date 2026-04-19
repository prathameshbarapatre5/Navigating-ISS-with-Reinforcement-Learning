import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# ISS 2D Occupancy Grid Map — Full 12-Module Layout
# ---------------------------------------------------------------------------

MAP_W = 160   # columns (x-axis)
MAP_H = 55    # rows    (y-axis)
CELL_SIZE = 0.5  # each cell = 0.5m

# ---------------------------------------------------------------------------
# Layout sketch (row=y, col=x, origin top-left)
#
#        [Poisk]   [Rassvet]
#           |          |
# [Nauka]-[Zvezda]-[Zarya]-[Unity]-[Destiny]-[Harmony]-[Columbus]
#                                     |                    |
#                               [Tranquility]          [Kibo ELM]
#                                     |                    |
#                                [Leonardo]            [Kibo PM]
#
# Main horizontal corridor: rows 23-31, full width
# Vertical branches off the main corridor
# ---------------------------------------------------------------------------

def build_iss_map():
    """
    Builds the 12-module ISS occupancy grid.

    Returns:
        grid (np.ndarray): shape (MAP_H, MAP_W), 0=free, 1=wall
        module_centers (dict): name -> (col, row) center of each module
    """
    grid = np.ones((MAP_H, MAP_W), dtype=np.int32)

    def carve(r1, c1, r2, c2):
        grid[r1:r2, c1:c2] = 0

 # MAIN HORIZONTAL CORRIDOR
    carve(23, 2, 31, 155)

    # --- RUSSIAN SEGMENT ---
    carve(21, 2,  33, 22)    # Nauka
    carve(21, 22, 33, 42)    # Zvezda
    carve(21, 42, 33, 62)    # Zarya

    # Poisk — UP off Zvezda (starts at row 8, goes DOWN to row 33 to overlap)
    carve(8, 26, 33, 38)

    # Rassvet — DOWN off Zarya (starts at row 21 to overlap with main corridor)
    carve(21, 46, 44, 58)

    # --- US / INTL SEGMENT ---
    carve(21, 62,  33, 82)   # Unity
    carve(21, 82,  33, 102)  # Destiny
    carve(21, 102, 33, 122)  # Harmony
    carve(21, 122, 33, 155)  # Columbus

    # Tranquility — DOWN off Unity
    carve(21, 66, 46, 78)

    # Leonardo — DOWN off Tranquility
    carve(44, 66, 54, 78)

    # Kibo ELM — UP off Harmony (goes all the way down to row 33 to overlap)
    carve(8, 106, 33, 118)

    # Kibo PM — UP above Kibo ELM (overlaps with ELM top)
    carve(2, 104, 10, 120)

    # Module centers (col, row) for goal sampling and labeling
    module_centers = {
            "nauka":       (12,  27),
            "zvezda":      (32,  27),
            "zarya":       (52,  27),
            "poisk":       (32,  16),
            "rassvet":     (52,  36),
            "unity":       (72,  27),
            "destiny":     (92,  27),
            "harmony":     (112, 27),
            "columbus":    (138, 27),
            "tranquility": (72,  36),
            "leonardo":    (72,  49),
            "kibo_elm":    (112, 17),
            "kibo_pm":     (112, 5),
        }

    return grid, module_centers


# Visualization
def render_map(grid, module_centers=None, agent_pos=None, goal_pos=None,
               title='ISSNav-v0: Full ISS Interior Map (12 Modules)'):

    fig, ax = plt.subplots(figsize=(18, 7))

    display = np.where(grid == 1, 0.15, 0.93)
    ax.imshow(display, cmap='gray', origin='upper',
              vmin=0, vmax=1, interpolation='nearest')

    if module_centers:
        for name, (col, row) in module_centers.items():
            ax.text(col, row, name.replace('_', '\n'),
                    ha='center', va='center',
                    fontsize=6, color='steelblue', fontweight='bold')

    if goal_pos is not None:
        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=13, zorder=5)

    if agent_pos is not None:
        ax.plot(agent_pos[0], agent_pos[1], 'go', markersize=10, zorder=5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#eeeeee', edgecolor='gray',
                       label='Free space'),
        mpatches.Patch(facecolor='#222222', label='Wall'),
    ]
    if agent_pos is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='g', markersize=10, label='Agent'))
    if goal_pos is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='r', markersize=13, label='Goal'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('X (cols)')
    ax.set_ylabel('Y (rows)')
    plt.tight_layout()
    plt.show()

# Quick test
if __name__ == "__main__":
    grid, module_centers = build_iss_map()

    print(f"Map shape     : {grid.shape}")
    print(f"Free cells    : {(grid == 0).sum()}")
    print(f"Wall cells    : {(grid == 1).sum()}")
    print(f"Modules       : {list(module_centers.keys())}")

    # Validate all module centers are in free space
    print("\nModule center check:")
    for name, (col, row) in module_centers.items():
        status = "FREE" if grid[row, col] == 0 else "WALL !!!"
        print(f"  {name:<14} ({col:>3}, {row:>2})  →  {status}")

    render_map(
        grid, module_centers,
        agent_pos=module_centers["nauka"],
        goal_pos=module_centers["columbus"]
    )