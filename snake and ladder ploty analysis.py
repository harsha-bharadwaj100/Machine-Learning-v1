import numpy as np
from scipy import linalg
import plotly.graph_objects as go
import plotly.subplots as sp
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


def print_transition_matrix(P, max_display=20):
    """
    Saves the whole transition matrix to a csv file and
    Print a readable subset of the transition matrix

    Parameters:
    P (numpy.ndarray): Transition matrix
    max_display (int): Maximum number of rows/columns to display
    """
    # Save the transition matrix to a csv file
    np.savetxt("transition_matrix.csv", P, delimiter=",")
    print("Transition matrix saved to 'transition_matrix.csv'")
    # Create labels for rows and columns
    labels = [str(i) for i in range(P.shape[0])]

    # If the matrix is too big, only show a subset
    if P.shape[0] > max_display:
        display_indices = list(range(0, min(10, max_display // 2))) + list(
            range(P.shape[0] - min(10, max_display // 2), P.shape[0])
        )
        sub_P = P[np.ix_(display_indices, display_indices)]
        sub_labels = [labels[i] for i in display_indices]
    else:
        sub_P = P
        sub_labels = labels

    # Create a pandas DataFrame for nice display
    df = pd.DataFrame(sub_P, index=sub_labels, columns=sub_labels)
    print("\nTransition Matrix (subset):")
    pd.set_option("display.precision", 3)
    print(df)

    # Reset pandas display options
    pd.reset_option("display.precision")


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


def plot_interactive_results(probabilities, exact_probabilities, game_params):
    """
    Create interactive plotly visualizations of the results.

    Parameters:
    probabilities (numpy.ndarray): Matrix of position probabilities for each step
    exact_probabilities (numpy.ndarray): Probability of finishing in exactly n moves
    game_params (dict): Dictionary containing game parameters for the plot title
    """
    n_steps = probabilities.shape[0] - 1
    final_position = probabilities.shape[1] - 1

    # Create subplots
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cumulative Probability of Reaching Final Position",
            "Probability of Reaching Final Position in Exactly n Moves",
            "Heatmap of Position Probabilities Over Time",
            "Top 10 Most Likely Positions After Various Moves",
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "heatmap"}, {"type": "xy"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # Plot 1: Cumulative probability line chart
    fig.add_trace(
        go.Scatter(
            x=list(range(n_steps + 1)),
            y=probabilities[:, final_position],
            mode="lines+markers",
            name="Cumulative Probability",
            line=dict(color="royalblue", width=3),
            hovertemplate="Moves: %{x}<br>Probability: %{y:.4f}",
        ),
        row=1,
        col=1,
    )

    # Add markers for key milestones (25%, 50%, 75%, 90%, 95%)
    milestones = [0.25, 0.50, 0.75, 0.90, 0.95]
    for milestone in milestones:
        # Find the first step that crosses the milestone
        for i, prob in enumerate(probabilities[:, final_position]):
            if prob >= milestone:
                fig.add_trace(
                    go.Scatter(
                        x=[i],
                        y=[prob],
                        mode="markers",
                        marker=dict(size=10, color="red"),
                        name=f"{milestone*100}% at move {i}",
                        hovertemplate=f"{milestone*100}% chance at move {i}",
                    ),
                    row=1,
                    col=1,
                )
                break

    # Plot 2: Probability of finishing in exactly n moves
    fig.add_trace(
        go.Bar(
            x=list(range(n_steps + 1)),
            y=exact_probabilities,
            name="Exact Probability",
            marker_color="mediumseagreen",
            hovertemplate="Moves: %{x}<br>Probability: %{y:.4f}",
        ),
        row=1,
        col=2,
    )

    # Plot 3: Heatmap of position probabilities over time
    # For clarity, we'll downsample the heatmap to show key positions
    step_interval = max(1, n_steps // 20)
    position_interval = max(1, final_position // 20)

    heatmap_steps = list(range(0, n_steps + 1, step_interval))
    heatmap_positions = list(range(0, final_position + 1, position_interval))

    heatmap_data = probabilities[heatmap_steps, :][:, heatmap_positions]

    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=[str(pos) for pos in heatmap_positions],
            y=[str(step) for step in heatmap_steps],
            colorscale="Viridis",
            colorbar=dict(title="Probability"),
            hovertemplate="Position: %{x}<br>Move: %{y}<br>Probability: %{z:.4f}",
        ),
        row=2,
        col=1,
    )

    # Plot 4: Top positions at different key steps
    key_steps = [5, 10, 15, 20, 25]
    key_steps = [step for step in key_steps if step <= n_steps]

    for step in key_steps:
        # Get the probability distribution for this step
        step_probs = probabilities[step, :]

        # Get the indices of the top 10 most likely positions (excluding the final position)
        non_final_probs = step_probs[:-1]  # Exclude the final position
        top_indices = np.argsort(non_final_probs)[-10:][::-1]

        # Add the final position as well
        positions = list(top_indices) + [final_position]
        probs = [step_probs[i] for i in positions]

        fig.add_trace(
            go.Bar(
                x=[str(pos) for pos in positions],
                y=probs,
                name=f"Move {step}",
                hovertemplate="Position: %{x}<br>Probability: %{y:.4f}",
            ),
            row=2,
            col=2,
        )

    # Configure layout
    fig.update_layout(
        title_text=f"Snake and Ladder Analysis: {game_params['title']}",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    # Update axes labels
    fig.update_xaxes(title_text="Number of Moves", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=1)

    fig.update_xaxes(title_text="Number of Moves", row=1, col=2)
    fig.update_yaxes(title_text="Probability", row=1, col=2)

    fig.update_xaxes(title_text="Position", row=2, col=1)
    fig.update_yaxes(title_text="Move Number", row=2, col=1)

    fig.update_xaxes(title_text="Position", row=2, col=2)
    fig.update_yaxes(title_text="Probability", row=2, col=2)

    return fig


def analyze_snake_ladder_game(
    snakes={},
    ladders={},
    final_position=100,
    start_position=0,
    n_steps=50,
    exact_roll_to_finish=False,
    print_matrix=True,
    interactive_plot=True,
):
    """
    Comprehensive analysis function that runs the entire snake and ladder analysis

    Parameters:
    snakes (dict): Dictionary mapping from snake heads to snake tails
    ladders (dict): Dictionary mapping from ladder bottoms to ladder tops
    final_position (int): The winning position
    start_position (int): The starting position
    n_steps (int): Number of moves to analyze
    exact_roll_to_finish (bool): Whether an exact roll is needed to finish
    print_matrix (bool): Whether to print the transition matrix
    interactive_plot (bool): Whether to create an interactive plot

    Returns:
    dict: Analysis results
    fig: Plotly figure if interactive_plot is True, else None
    """
    # Create title for plots
    rule_text = (
        "Exact roll required to finish" if exact_roll_to_finish else "Standard rules"
    )
    game_params = {
        "title": f"{len(snakes)} Snakes, {len(ladders)} Ladders, {rule_text}",
        "snakes": len(snakes),
        "ladders": len(ladders),
        "exact_roll": exact_roll_to_finish,
    }

    # Create the transition matrix
    P = create_transition_matrix(snakes, ladders, final_position, exact_roll_to_finish)

    # Print the transition matrix if requested
    if print_matrix:
        print_transition_matrix(P)

    # Calculate position probabilities for each step
    probabilities = calculate_position_probabilities(P, start_position, n_steps)

    # Calculate probability of finishing in exactly n moves
    exact_probabilities = probability_of_finishing_in_exactly_n_moves(probabilities)

    # Calculate expected number of moves
    expected_moves = expected_number_of_moves(P, start_position, final_position)

    # Create interactive plot if requested
    fig = None
    if interactive_plot:
        fig = plot_interactive_results(probabilities, exact_probabilities, game_params)

    # Print results
    print(f"\nGame Analysis: {game_params['title']}")
    print(f"Expected number of moves to finish the game: {expected_moves:.2f}")

    print("\nProbability of finishing in exactly n moves:")
    significant_moves = [
        (i, prob) for i, prob in enumerate(exact_probabilities) if prob > 0.005
    ]
    for i, prob in significant_moves:
        print(f"n = {i}: {prob:.6f}")

    print("\nCumulative probability of finishing within n moves:")
    for i in range(0, n_steps + 1, 5):  # Show every 5th step
        print(f"n ≤ {i}: {probabilities[i, -1]:.6f}")

    # Create results dictionary
    results = {
        "expected_moves": expected_moves,
        "exact_probabilities": exact_probabilities,
        "cumulative_probabilities": probabilities[:, -1],
        "position_probabilities": probabilities,
        "game_parameters": game_params,
    }

    return results, fig


def compare_game_variants(variants, n_steps=50):
    """
    Compare multiple variants of the snake and ladder game

    Parameters:
    variants (list): List of dictionaries, each containing parameters for a game variant
    n_steps (int): Number of moves to analyze

    Returns:
    fig: Plotly figure comparing the variants
    """
    # Create figure
    fig = go.Figure()

    # Analyze each variant
    for variant in variants:
        # Extract parameters
        snakes = variant.get("snakes", {})
        ladders = variant.get("ladders", {})
        exact_roll = variant.get("exact_roll_to_finish", False)
        name = variant.get("name", "Unnamed Variant")

        # Create transition matrix
        P = create_transition_matrix(snakes, ladders, 100, exact_roll)

        # Calculate probabilities
        probabilities = calculate_position_probabilities(P, 0, n_steps)

        # Add to plot
        fig.add_trace(
            go.Scatter(
                x=list(range(n_steps + 1)),
                y=probabilities[:, -1],
                mode="lines",
                name=name,
                hovertemplate="Moves: %{x}<br>Probability: %{y:.4f}",
            )
        )

    # Configure layout
    fig.update_layout(
        title="Comparison of Snake and Ladder Game Variants",
        xaxis_title="Number of Moves",
        yaxis_title="Cumulative Probability of Finishing",
        legend_title="Game Variant",
        template="plotly_white",
    )

    return fig


# Standard game configuration
standard_snakes = {51: 11, 56: 15, 62: 57, 92: 53, 98: 8}

standard_ladders = {2: 38, 4: 14, 9: 31, 33: 85, 52: 88, 80: 99}

# Example usage
if __name__ == "__main__":
    # Analyze standard game with both rule variants
    print("\n--- STANDARD RULES ANALYSIS ---")
    results1, fig1 = analyze_snake_ladder_game(
        snakes=standard_snakes,
        ladders=standard_ladders,
        exact_roll_to_finish=False,
        n_steps=100,
    )

    print("\n--- EXACT ROLL TO FINISH ANALYSIS ---")
    results2, fig2 = analyze_snake_ladder_game(
        snakes=standard_snakes,
        ladders=standard_ladders,
        exact_roll_to_finish=True,
        n_steps=100,
    )

    # Compare the two variants
    variants = [
        {
            "name": "Standard Rules",
            "snakes": standard_snakes,
            "ladders": standard_ladders,
            "exact_roll_to_finish": False,
        },
        {
            "name": "Exact Roll to Finish",
            "snakes": standard_snakes,
            "ladders": standard_ladders,
            "exact_roll_to_finish": True,
        },
    ]

    comparison_fig = compare_game_variants(variants)

    # Show the plots (in a Jupyter notebook these will display automatically)
    fig1.show()
    fig2.show()
    comparison_fig.show()

    print("\nRun the following functions to create your own game variants:")
    print("1. analyze_snake_ladder_game() - Analyze a single game configuration")
    print("2. compare_game_variants() - Compare multiple game configurations")
