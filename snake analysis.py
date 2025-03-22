import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_transition_matrix(
    snakes, ladders, final_position=100, exact_roll_to_finish=False
):
    """
    Create the transition matrix for a snake and ladder game.

    Parameters:
    snakes (dict): Dictionary mapping from snake heads to snake tails
    ladders (dict): Dictionary mapping from ladder bottoms to ladder tops
    final_position (int): The winning position (default 100)
    exact_roll_to_finish (bool): If True, players must roll exactly the right number to reach the final position

    Returns:
    numpy.ndarray: Transition probability matrix
    """
    # Create a matrix of size (final_position+1) x (final_position+1)
    # We include position 0 as the starting position
    size = final_position + 1
    P = np.zeros((size, size))

    # For each position (except the final one)
    for i in range(size - 1):
        # For each possible roll of the die
        for roll in range(1, 7):
            # Calculate the next position
            next_pos = i + roll

            # Handle the exact_roll_to_finish rule
            if exact_roll_to_finish and i > final_position - 6:
                # If we're within 6 spaces of the final position
                if next_pos > final_position:
                    # Stay in place if we would exceed the final position
                    next_pos = i
                elif next_pos == final_position:
                    # Allow exact rolls to the final position
                    pass
            else:
                # Standard rule: if we exceed the final position, stay where we are
                if next_pos > final_position:
                    next_pos = i

            # Check if we landed on a snake or ladder
            if next_pos in snakes:
                next_pos = snakes[next_pos]
            elif next_pos in ladders:
                next_pos = ladders[next_pos]

            # Update transition probability
            P[i, next_pos] += 1 / 6

    # Once at the final position, stay there (absorbing state)
    P[final_position, final_position] = 1

    return P


def compare_transition_matrices(
    P1, P2, names=("Board 1", "Board 2"), title="Transition Matrix Comparison"
):
    """
    Compare two transition matrices and visualize their differences

    Parameters:
    P1 (numpy.ndarray): First transition matrix
    P2 (numpy.ndarray): Second transition matrix
    names (tuple): Names for the two boards
    title (str): Title for the plots
    """
    # Calculate the difference matrix
    diff_matrix = P1 - P2

    # Create a figure with 3 subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Transition Matrix: {names[0]}",
            f"Transition Matrix: {names[1]}",
            "Difference Matrix",
            "Key Transition Differences",
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "bar"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    # Add heatmaps for the transition matrices
    fig.add_trace(
        go.Heatmap(
            z=P1,
            colorscale="Blues",
            showscale=False,
            hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{z:.4f}",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=P2,
            colorscale="Greens",
            showscale=False,
            hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{z:.4f}",
        ),
        row=1,
        col=2,
    )

    # Add heatmap for the difference matrix
    fig.add_trace(
        go.Heatmap(
            z=diff_matrix,
            colorscale="RdBu",
            zmid=0,
            hovertemplate="From: %{y}<br>To: %{x}<br>Diff: %{z:.4f}",
        ),
        row=2,
        col=1,
    )

    # Find the top differences for the bar chart
    # Flatten the matrices and find the indices with the largest absolute differences
    flat_diff = diff_matrix.flatten()
    top_indices = np.argsort(np.abs(flat_diff))[-20:]  # Get top 20 differences

    # Convert flat indices back to 2D
    size = P1.shape[0]
    from_positions = [idx // size for idx in top_indices]
    to_positions = [idx % size for idx in top_indices]

    # Get the differences and format labels
    top_diffs = [flat_diff[idx] for idx in top_indices]
    labels = [
        f"{from_pos}->{to_pos}"
        for from_pos, to_pos in zip(from_positions, to_positions)
    ]

    # Sort by the actual difference value for better visualization
    sorted_indices = np.argsort(top_diffs)
    labels = [labels[i] for i in sorted_indices]
    top_diffs = [top_diffs[i] for i in sorted_indices]

    # Add horizontal bar chart for top differences
    fig.add_trace(
        go.Bar(
            y=labels,
            x=top_diffs,
            orientation="h",
            marker=dict(
                color=top_diffs,
                colorscale="RdBu",
                cmin=-max(abs(min(top_diffs)), abs(max(top_diffs))),
                cmax=max(abs(min(top_diffs)), abs(max(top_diffs))),
            ),
            hovertemplate="Transition: %{y}<br>Difference: %{x:.4f}",
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(title_text=title, height=900, width=1100, template="plotly_white")

    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            if i == 2 and j == 2:
                # For the bar chart
                fig.update_xaxes(title_text="Probability Difference", row=i, col=j)
                fig.update_yaxes(title_text="Transition", row=i, col=j)
            else:
                # For the heatmaps
                fig.update_xaxes(title_text="To Position", row=i, col=j)
                fig.update_yaxes(title_text="From Position", row=i, col=j)

    return fig


def analyze_key_differences(
    P1, P2, snakes1, ladders1, snakes2, ladders2, names=("Board 1", "Board 2")
):
    """
    Analyze and explain key differences between two snake and ladder boards

    Parameters:
    P1, P2 (numpy.ndarray): Transition matrices
    snakes1, snakes2 (dict): Snake configurations
    ladders1, ladders2 (dict): Ladder configurations
    names (tuple): Names for the two boards
    """
    # Calculate the difference matrix
    diff_matrix = P1 - P2

    # Find positions with significant differences
    threshold = 0.1  # Adjust this threshold as needed
    significant_diffs = np.where(np.abs(diff_matrix) > threshold)

    # Print analysis
    print(f"\nKey Differences Between {names[0]} and {names[1]}:")
    print("=" * 50)

    # Analyze snake differences
    print("\nSnake Configuration Differences:")
    only_in_1 = {k: v for k, v in snakes1.items() if k not in snakes2}
    only_in_2 = {k: v for k, v in snakes2.items() if k not in snakes1}
    common_diff = {
        k: (v, snakes2[k])
        for k, v in snakes1.items()
        if k in snakes2 and v != snakes2[k]
    }

    if only_in_1:
        print(f"Snakes only in {names[0]}: {only_in_1}")
    if only_in_2:
        print(f"Snakes only in {names[1]}: {only_in_2}")
    if common_diff:
        print(f"Snakes with different endpoints: {common_diff}")

    # Analyze ladder differences
    print("\nLadder Configuration Differences:")
    only_in_1 = {k: v for k, v in ladders1.items() if k not in ladders2}
    only_in_2 = {k: v for k, v in ladders2.items() if k not in ladders1}
    common_diff = {
        k: (v, ladders2[k])
        for k, v in ladders1.items()
        if k in ladders2 and v != ladders2[k]
    }

    if only_in_1:
        print(f"Ladders only in {names[0]}: {only_in_1}")
    if only_in_2:
        print(f"Ladders only in {names[1]}: {only_in_2}")
    if common_diff:
        print(f"Ladders with different endpoints: {common_diff}")

    # Analyze significant transition differences
    if len(significant_diffs[0]) > 0:
        print("\nTop Significant Transition Probability Differences:")

        # Create a list of (from, to, diff) tuples
        diffs = [
            (
                significant_diffs[0][i],
                significant_diffs[1][i],
                diff_matrix[significant_diffs[0][i], significant_diffs[1][i]],
            )
            for i in range(len(significant_diffs[0]))
        ]

        # Sort by absolute difference
        diffs.sort(key=lambda x: abs(x[2]), reverse=True)

        # Print top 10 differences (or fewer if there aren't that many)
        for i in range(min(10, len(diffs))):
            from_pos, to_pos, diff = diffs[i]
            print(f"Transition {from_pos}->{to_pos}: Difference of {diff:.4f}")

            # Explain the difference
            if abs(diff) > 0:
                if diff > 0:
                    higher = names[0]
                    lower = names[1]
                else:
                    higher = names[1]
                    lower = names[0]
                    diff = -diff

                print(f"  More likely in {higher} than in {lower} by {diff:.4f}")

                # Try to explain why
                if from_pos in snakes1 and from_pos not in snakes2:
                    print(
                        f"  Possible reason: Position {from_pos} has a snake in {names[0]} but not in {names[1]}"
                    )
                elif from_pos in snakes2 and from_pos not in snakes1:
                    print(
                        f"  Possible reason: Position {from_pos} has a snake in {names[1]} but not in {names[0]}"
                    )
                elif from_pos in ladders1 and from_pos not in ladders2:
                    print(
                        f"  Possible reason: Position {from_pos} has a ladder in {names[0]} but not in {names[1]}"
                    )
                elif from_pos in ladders2 and from_pos not in ladders1:
                    print(
                        f"  Possible reason: Position {from_pos} has a ladder in {names[1]} but not in {names[0]}"
                    )

                # Check if to_pos is reachable by dice roll
                if 1 <= to_pos - from_pos <= 6:
                    roll_needed = to_pos - from_pos
                    print(f"  This transition happens with a roll of {roll_needed}")

    # Calculate expected number of moves for both boards
    Q1 = P1[:100, :100]
    Q2 = P2[:100, :100]

    try:
        N1 = np.linalg.inv(np.eye(100) - Q1)
        N2 = np.linalg.inv(np.eye(100) - Q2)

        expected_moves1 = np.sum(N1[0, :])
        expected_moves2 = np.sum(N2[0, :])

        print(
            f"\nExpected number of moves to finish in {names[0]}: {expected_moves1:.2f}"
        )
        print(
            f"Expected number of moves to finish in {names[1]}: {expected_moves2:.2f}"
        )
        print(f"Difference in expected moves: {expected_moves1 - expected_moves2:.2f}")
    except:
        print("\nCouldn't calculate expected moves (singular matrix issue)")


def visualize_board_differences(
    snakes1, ladders1, snakes2, ladders2, names=("Board 1", "Board 2")
):
    """
    Create a visual representation of the two different boards

    Parameters:
    snakes1, snakes2 (dict): Snake configurations
    ladders1, ladders2 (dict): Ladder configurations
    names (tuple): Names for the two boards
    """
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Set up the axes for both boards
    ax1.set_title(f"{names[0]} Configuration", fontsize=16)
    ax2.set_title(f"{names[1]} Configuration", fontsize=16)

    for ax in [ax1, ax2]:
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlabel("Position", fontsize=12)
        ax.set_ylabel("Position", fontsize=12)

    # Draw snakes and ladders for board 1
    for start, end in snakes1.items():
        ax1.plot([start, end], [start, end], "r-", linewidth=2, alpha=0.7)
        ax1.plot(start, start, "ro", markersize=8)
        ax1.plot(end, end, "rs", markersize=8)
        ax1.annotate(
            f"{start}→{end}",
            xy=(start, start),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    for start, end in ladders1.items():
        ax1.plot([start, end], [start, end], "g-", linewidth=2, alpha=0.7)
        ax1.plot(start, start, "go", markersize=8)
        ax1.plot(end, end, "g^", markersize=8)
        ax1.annotate(
            f"{start}→{end}",
            xy=(start, start),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    # Draw snakes and ladders for board 2
    for start, end in snakes2.items():
        ax2.plot([start, end], [start, end], "r-", linewidth=2, alpha=0.7)
        ax2.plot(start, start, "ro", markersize=8)
        ax2.plot(end, end, "rs", markersize=8)
        ax2.annotate(
            f"{start}→{end}",
            xy=(start, start),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    for start, end in ladders2.items():
        ax2.plot([start, end], [start, end], "g-", linewidth=2, alpha=0.7)
        ax2.plot(start, start, "go", markersize=8)
        ax2.plot(end, end, "g^", markersize=8)
        ax2.annotate(
            f"{start}→{end}",
            xy=(start, start),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    # Add a legend
    ax1.plot([], [], "r-", label="Snake")
    ax1.plot([], [], "ro", label="Snake Head")
    ax1.plot([], [], "rs", label="Snake Tail")
    ax1.plot([], [], "g-", label="Ladder")
    ax1.plot([], [], "go", label="Ladder Bottom")
    ax1.plot([], [], "g^", label="Ladder Top")
    ax1.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    return fig


def analyze_games(custom_snakes, custom_ladders, standard_snakes, standard_ladders):
    """
    Compare two Snake and Ladder game configurations

    Parameters:
    custom_snakes, custom_ladders: Custom board configuration
    standard_snakes, standard_ladders: Standard board configuration
    """
    # Create transition matrices
    P_custom = create_transition_matrix(custom_snakes, custom_ladders)
    P_standard = create_transition_matrix(standard_snakes, standard_ladders)

    # Compare transition matrices
    fig1 = compare_transition_matrices(
        P_custom,
        P_standard,
        names=("Custom Board", "Standard Board"),
        title="Snake and Ladder Transition Matrix Comparison",
    )

    # Analyze key differences
    analyze_key_differences(
        P_custom,
        P_standard,
        custom_snakes,
        custom_ladders,
        standard_snakes,
        standard_ladders,
        names=("Custom Board", "Standard Board"),
    )

    # Visualize board differences
    fig2 = visualize_board_differences(
        custom_snakes,
        custom_ladders,
        standard_snakes,
        standard_ladders,
        names=("Custom Board", "Standard Board"),
    )

    return fig1, fig2


# Define board configurations
custom_snakes = {51: 11, 56: 15, 62: 57, 92: 53, 98: 8}
custom_ladders = {2: 38, 4: 14, 9: 31, 33: 85, 52: 88, 80: 99}

standard_snakes = {
    16: 6,
    47: 26,
    49: 11,
    56: 53,
    62: 19,
    64: 60,
    87: 24,
    93: 73,
    95: 75,
    98: 78,
}

standard_ladders = {
    1: 38,
    4: 14,
    9: 31,
    21: 42,
    28: 84,
    36: 44,
    51: 67,
    71: 91,
    80: 100,
}

# Run the analysis
transition_fig, board_fig = analyze_games(
    custom_snakes, custom_ladders, standard_snakes, standard_ladders
)

# Show the figures
transition_fig.show()
# In Jupyter notebook, the matplotlib figure would show automatically
plt.show()

print("\nAnalysis Complete!")
