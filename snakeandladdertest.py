import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd


def create_transition_matrix(
    snakes, ladders, final_position=100, exact_roll_to_finish=False
):
    """
    Create the transition matrix for a snake and ladder game.

    Parameters:
    snakes (dict): Dictionary mapping from snake heads to snake tails
    ladders (dict): Dictionary mapping from ladder bottoms to ladder tops
    final_position (int): The winning position (default 100)
    exact_roll_to_finish (bool): If True, player must roll exactly to reach final position

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

            # Handle the exact roll to finish rule
            if exact_roll_to_finish and i > final_position - 6:
                if next_pos == final_position:
                    # Perfect roll to reach the final position
                    pass
                elif next_pos > final_position:
                    # Overshot, stay at current position
                    next_pos = i
            # If we exceed the final position without the exact rule, stay where we are
            elif next_pos > final_position:
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


def print_transition_matrix(P, max_rows=10, max_cols=10):
    """
    Print a portion of the transition matrix for visualization.

    Parameters:
    P (numpy.ndarray): Transition matrix
    max_rows (int): Maximum number of rows to print
    max_cols (int): Maximum number of columns to print
    """
    rows = min(max_rows, P.shape[0])
    cols = min(max_cols, P.shape[1])

    print(f"\nPartial Transition Matrix ({rows}x{cols} of {P.shape[0]}x{P.shape[1]}):")
    print("-" * 50)

    # Print column headers
    print("    ", end="")
    for j in range(cols):
        print(f"{j:4d}", end="")
    print("\n    ", end="")
    print("-" * (4 * cols))

    # Print matrix rows
    for i in range(rows):
        print(f"{i:2d} |", end="")
        for j in range(cols):
            print(f"{P[i, j]:4.2f}", end="")
        print()

    print("-" * 50)

    # Show a few specific important transitions
    print("\nSelected transitions:")
    examples = []
    # Show some snake transitions
    for head in list(snakes.keys())[:3]:
        examples.append((head, snakes[head]))
    # Show some ladder transitions
    for bottom in list(ladders.keys())[:3]:
        examples.append((bottom, ladders[bottom]))

    for start, end in examples:
        if start < P.shape[0] and end < P.shape[1]:
            probs = [
                P[start - roll, start] for roll in range(1, 7) if start - roll >= 0
            ]
            if probs:
                avg_prob = sum(probs) / len(probs)
                print(
                    f"Transition {start} → {end} (Snake/Ladder): Avg. incoming prob = {avg_prob:.4f}"
                )


def calculate_position_probabilities(P, start_position=0, n_steps=20):
    """
    Calculate the probability of being at each position after n steps.

    Parameters:
    P (numpy.ndarray): Transition matrix
    start_position (int): Starting position
    n_steps (int): Number of steps

    Returns:
    numpy.ndarray: Probability distribution over positions
    """
    # Create initial state vector
    v = np.zeros(P.shape[0])
    v[start_position] = 1

    # Store probabilities for each step
    probabilities = np.zeros((n_steps + 1, P.shape[0]))
    probabilities[0] = v

    # Calculate probabilities for each step
    for i in range(1, n_steps + 1):
        v = v @ P
        probabilities[i] = v

    return probabilities


def probability_of_finishing_in_exactly_n_moves(probabilities):
    """
    Calculate the probability of finishing in exactly n moves.

    Parameters:
    probabilities (numpy.ndarray): Matrix of position probabilities for each step

    Returns:
    numpy.ndarray: Probability of finishing in exactly n moves
    """
    n_steps = probabilities.shape[0] - 1
    final_position = probabilities.shape[1] - 1

    exact_prob = np.zeros(n_steps + 1)
    exact_prob[0] = probabilities[0, final_position]

    for i in range(1, n_steps + 1):
        exact_prob[i] = (
            probabilities[i, final_position] - probabilities[i - 1, final_position]
        )

    return exact_prob


def plot_interactive_results(
    probabilities, exact_probabilities, P, snakes, ladders, exact_roll_rule
):
    """
    Create interactive and informative plots for analyzing the results.

    Parameters:
    probabilities (numpy.ndarray): Matrix of position probabilities for each step
    exact_probabilities (numpy.ndarray): Probability of finishing in exactly n moves
    P (numpy.ndarray): Transition matrix
    snakes (dict): Dictionary of snakes
    ladders (dict): Dictionary of ladders
    exact_roll_rule (bool): Whether exact roll to finish rule is enabled
    """
    n_steps = probabilities.shape[0] - 1
    final_position = probabilities.shape[1] - 1

    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f'Snake and Ladder Game Analysis {"(Exact Roll to Finish)" if exact_roll_rule else ""}',
        fontsize=16,
        fontweight="bold",
    )

    # 1. Cumulative probability plot
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    (cumul_line,) = ax1.plot(
        range(n_steps + 1),
        probabilities[:, final_position],
        "b-",
        marker="o",
        linewidth=2,
    )
    ax1.set_title("Cumulative Probability of Reaching Position 100")
    ax1.set_xlabel("Number of Moves")
    ax1.set_ylabel("Probability")
    ax1.grid(True)

    # Add threshold lines for key probabilities
    thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
    colors = ["r", "g", "purple", "orange", "brown"]

    for threshold, color in zip(thresholds, colors):
        # Find the first move where probability exceeds threshold
        crossing_move = np.where(probabilities[:, final_position] >= threshold)[0]
        if len(crossing_move) > 0:
            cross_move = crossing_move[0]
            ax1.axhline(y=threshold, color=color, linestyle="--", alpha=0.5)
            ax1.axvline(x=cross_move, color=color, linestyle="--", alpha=0.5)
            ax1.text(
                cross_move + 0.5,
                threshold - 0.03,
                f"{cross_move} moves",
                bbox=dict(facecolor="white", alpha=0.7),
            )

    # 2. Probability of finishing in exactly n moves
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    bars = ax2.bar(range(n_steps + 1), exact_probabilities, color="skyblue")
    ax2.set_title("Probability of Reaching Position 100 in Exactly n Moves")
    ax2.set_xlabel("Number of Moves")
    ax2.set_ylabel("Probability")
    ax2.grid(True, axis="y")

    # Find and highlight the most likely number of moves
    most_likely_moves = np.argmax(exact_probabilities)
    bars[most_likely_moves].set_color("red")
    ax2.annotate(
        f"Most likely: {most_likely_moves} moves\n({exact_probabilities[most_likely_moves]:.4f})",
        xy=(most_likely_moves, exact_probabilities[most_likely_moves]),
        xytext=(most_likely_moves + 5, exact_probabilities[most_likely_moves] + 0.01),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
    )

    # 3. Heat map of position probabilities over time
    ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    # Extract positions with significant probabilities
    position_subset = list(range(0, 101, 5))  # Sample positions every 5 steps
    prob_matrix = probabilities[:, position_subset]

    # Create a heatmap
    sns.heatmap(
        prob_matrix.T,
        ax=ax3,
        cmap="viridis",
        norm=LogNorm(),
        xticklabels=5,
        yticklabels=position_subset,
        cbar_kws={"label": "Probability (log scale)"},
    )
    ax3.set_title("Position Probabilities Over Time")
    ax3.set_xlabel("Number of Moves")
    ax3.set_ylabel("Position")

    # 4. Game board visualization with snakes and ladders
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    visualize_game_board(ax4, snakes, ladders, P, final_position)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    return fig


def visualize_game_board(ax, snakes, ladders, P, final_position):
    """
    Visualize the game board with snakes, ladders, and transition probabilities.

    Parameters:
    ax (matplotlib.axes.Axes): Axes to plot on
    snakes (dict): Dictionary of snakes
    ladders (dict): Dictionary of ladders
    P (numpy.ndarray): Transition matrix
    final_position (int): Final position on the board
    """
    # Create a 10x10 grid for the board
    board_size = 10
    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)

    # Function to convert position to board coordinates
    def pos_to_coords(pos):
        if pos == 0:  # Starting position
            return (-0.5, -0.5)

        pos -= 1  # Adjust for 0-indexing
        row = 9 - (pos // 10)

        # Alternate direction based on row
        if (9 - row) % 2 == 0:  # Even rows go left to right
            col = pos % 10
        else:  # Odd rows go right to left
            col = 9 - (pos % 10)

        return (col, row)

    # Draw grid
    for i in range(board_size + 1):
        ax.axhline(y=i, color="gray", linestyle="-", alpha=0.3)
        ax.axvline(x=i, color="gray", linestyle="-", alpha=0.3)

    # Place position numbers
    for pos in range(1, final_position + 1):
        col, row = pos_to_coords(pos)
        ax.text(
            col + 0.5,
            row + 0.5,
            str(pos),
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # Draw snakes
    for head, tail in snakes.items():
        head_x, head_y = pos_to_coords(head)
        tail_x, tail_y = pos_to_coords(tail)
        ax.plot(
            [head_x + 0.5, tail_x + 0.5],
            [head_y + 0.5, tail_y + 0.5],
            "r-",
            linewidth=2.5,
            alpha=0.7,
        )
        ax.plot(head_x + 0.5, head_y + 0.5, "ro", markersize=8)  # Snake head
        ax.plot(tail_x + 0.5, tail_y + 0.5, "ro", markersize=6)  # Snake tail

    # Draw ladders
    for bottom, top in ladders.items():
        bottom_x, bottom_y = pos_to_coords(bottom)
        top_x, top_y = pos_to_coords(top)
        ax.plot(
            [bottom_x + 0.5, top_x + 0.5],
            [bottom_y + 0.5, top_y + 0.5],
            "g-",
            linewidth=2.5,
            alpha=0.7,
        )
        ax.plot(bottom_x + 0.5, bottom_y + 0.5, "gs", markersize=6)  # Ladder bottom
        ax.plot(top_x + 0.5, top_y + 0.5, "gs", markersize=8)  # Ladder top

    # Highlight start and finish
    start_x, start_y = pos_to_coords(1)
    finish_x, finish_y = pos_to_coords(final_position)
    ax.plot(start_x + 0.5, start_y + 0.5, "go", markersize=10, alpha=0.8)  # Start
    ax.plot(finish_x + 0.5, finish_y + 0.5, "bo", markersize=10, alpha=0.8)  # Finish

    # Add a legend
    ax.plot([], [], "r-", linewidth=2.5, label="Snake")
    ax.plot([], [], "g-", linewidth=2.5, label="Ladder")
    ax.plot([], [], "go", markersize=10, label="Start")
    ax.plot([], [], "bo", markersize=10, label="Finish")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)

    ax.set_title("Game Board with Snakes and Ladders")
    ax.set_xticks([])
    ax.set_yticks([])


def expected_number_of_moves(P, start_position=0, final_position=100):
    """
    Calculate the expected number of moves to reach the final position.

    Parameters:
    P (numpy.ndarray): Transition matrix
    start_position (int): Starting position
    final_position (int): Final position

    Returns:
    float: Expected number of moves
    """
    # Remove the absorbing state (final position)
    Q = P[:final_position, :final_position]

    # Calculate the fundamental matrix
    N = linalg.inv(np.eye(final_position) - Q)

    # The expected number of moves is the sum of the row corresponding to the starting position
    expected_moves = np.sum(N[start_position, :])

    return expected_moves


def analyze_rules_comparison(snakes, ladders, n_steps=50):
    """
    Compare the game with and without the exact roll to finish rule.

    Parameters:
    snakes (dict): Dictionary of snakes
    ladders (dict): Dictionary of ladders
    n_steps (int): Number of steps to analyze

    Returns:
    tuple: Figures for each rule and comparison data
    """
    # Analyze with normal rules
    P_normal = create_transition_matrix(snakes, ladders, exact_roll_to_finish=False)
    prob_normal = calculate_position_probabilities(
        P_normal, start_position=0, n_steps=n_steps
    )
    exact_prob_normal = probability_of_finishing_in_exactly_n_moves(prob_normal)
    expected_moves_normal = expected_number_of_moves(P_normal)

    # Analyze with exact roll to finish rule
    P_exact = create_transition_matrix(snakes, ladders, exact_roll_to_finish=True)
    prob_exact = calculate_position_probabilities(
        P_exact, start_position=0, n_steps=n_steps
    )
    exact_prob_exact = probability_of_finishing_in_exactly_n_moves(prob_exact)
    expected_moves_exact = expected_number_of_moves(P_exact)

    # Create individual plots
    fig_normal = plot_interactive_results(
        prob_normal, exact_prob_normal, P_normal, snakes, ladders, False
    )
    fig_exact = plot_interactive_results(
        prob_exact, exact_prob_exact, P_exact, snakes, ladders, True
    )

    # Create comparison figure
    fig_compare = plt.figure(figsize=(14, 8))
    plt.suptitle(
        "Comparison: Normal Rules vs. Exact Roll to Finish",
        fontsize=16,
        fontweight="bold",
    )

    # Compare cumulative probabilities
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(
        range(n_steps + 1), prob_normal[:, -1], "b-", label="Normal Rules", linewidth=2
    )
    ax1.plot(
        range(n_steps + 1),
        prob_exact[:, -1],
        "r--",
        label="Exact Roll to Finish",
        linewidth=2,
    )
    ax1.set_title("Cumulative Probability of Finishing")
    ax1.set_xlabel("Number of Moves")
    ax1.set_ylabel("Probability")
    ax1.grid(True)
    ax1.legend()

    # Find moves needed for 50%, 90%, 99% probability
    thresholds = [0.5, 0.9, 0.99]
    for threshold in thresholds:
        normal_moves = (
            np.where(prob_normal[:, -1] >= threshold)[0][0]
            if any(prob_normal[:, -1] >= threshold)
            else np.inf
        )
        exact_moves = (
            np.where(prob_exact[:, -1] >= threshold)[0][0]
            if any(prob_exact[:, -1] >= threshold)
            else np.inf
        )
        ax1.annotate(
            f"{threshold*100:.0f}%: {normal_moves} vs {exact_moves} moves",
            xy=(max(normal_moves, exact_moves) + 2, threshold),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    # Compare exact probabilities
    ax2 = plt.subplot(2, 1, 2)
    x = np.arange(len(exact_prob_normal))
    width = 0.35
    ax2.bar(
        x - width / 2, exact_prob_normal, width, label="Normal Rules", color="skyblue"
    )
    ax2.bar(
        x + width / 2,
        exact_prob_exact,
        width,
        label="Exact Roll to Finish",
        color="salmon",
    )
    ax2.set_title("Probability of Finishing in Exactly n Moves")
    ax2.set_xlabel("Number of Moves")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.grid(True, axis="y")

    # Add text with expected moves
    plt.figtext(
        0.5,
        0.01,
        f"Expected Moves - Normal Rules: {expected_moves_normal:.2f} | Exact Roll to Finish: {expected_moves_exact:.2f}",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Print summary comparison
    print("\n===== RULE COMPARISON =====")
    print(f"Expected moves (Normal Rules): {expected_moves_normal:.2f}")
    print(f"Expected moves (Exact Roll): {expected_moves_exact:.2f}")
    print(f"Difference: {expected_moves_exact - expected_moves_normal:.2f} moves")

    # Find the most likely number of moves
    most_likely_normal = np.argmax(exact_prob_normal)
    most_likely_exact = np.argmax(exact_prob_exact)
    print(
        f"Most likely # of moves (Normal): {most_likely_normal} ({exact_prob_normal[most_likely_normal]:.4f})"
    )
    print(
        f"Most likely # of moves (Exact): {most_likely_exact} ({exact_prob_exact[most_likely_exact]:.4f})"
    )

    # Find moves to reach certain probabilities
    for prob in [0.5, 0.75, 0.9, 0.95]:
        normal_idx = next(
            (i for i, p in enumerate(prob_normal[:, -1]) if p >= prob), None
        )
        exact_idx = next(
            (i for i, p in enumerate(prob_exact[:, -1]) if p >= prob), None
        )
        print(
            f"Moves for {prob*100:.0f}% probability - Normal: {normal_idx}, Exact: {exact_idx}"
        )

    return (
        fig_normal,
        fig_exact,
        fig_compare,
        {
            "expected_normal": expected_moves_normal,
            "expected_exact": expected_moves_exact,
            "most_likely_normal": most_likely_normal,
            "most_likely_exact": most_likely_exact,
        },
    )


# Main execution
if __name__ == "__main__":
    # Define some snakes and ladders for a standard game
    snakes = {
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

    ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}

    # Print game configuration
    print("=== SNAKES AND LADDERS GAME ANALYSIS ===")
    print(f"Number of snakes: {len(snakes)}")
    print(f"Number of ladders: {len(ladders)}")

    # Create transition matrices for both rule sets
    P_normal = create_transition_matrix(snakes, ladders, exact_roll_to_finish=False)
    P_exact = create_transition_matrix(snakes, ladders, exact_roll_to_finish=True)

    # Print transition matrices
    print_transition_matrix(P_normal, max_rows=12, max_cols=12)
    print("\nWith Exact Roll to Finish rule:")
    # Just show the relevant portion that differs
    print_transition_matrix(P_exact[94:100, 94:101], max_rows=6, max_cols=7)

    # Do full analysis and comparison
    n_steps = 50  # Number of moves to consider
    fig_normal, fig_exact, fig_compare, stats = analyze_rules_comparison(
        snakes, ladders, n_steps
    )

    # Save figures if needed
    # fig_normal.savefig('normal_rules.png', dpi=300, bbox_inches='tight')
    # fig_exact.savefig('exact_roll_rules.png', dpi=300, bbox_inches='tight')
    # fig_compare.savefig('rules_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()
