import re
import time
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(page_title="DFA Simulator", layout="wide")

# Initialize session state
if 'dfas' not in st.session_state:
    st.session_state.dfas = {}
if 'current_dfa' not in st.session_state:
    st.session_state.current_dfa = None
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'auto_simulate' not in st.session_state:
    st.session_state.auto_simulate = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0


class DFA:
    def __init__(self, name):
        self.name = name
        self.states = set()
        self.alphabet = set()
        self.transitions = {}
        self.start_state = None
        self.accept_states = set()

    def add_state(self, state, is_start=False, is_accept=False):
        self.states.add(state)
        if is_start:
            self.start_state = state
        if is_accept:
            self.accept_states.add(state)

    def add_transition(self, from_state, symbol, to_state):
        if from_state not in self.states or to_state not in self.states:
            return False
        if symbol not in self.alphabet:
            return False
        self.transitions[(from_state, symbol)] = to_state
        return True

    def is_valid(self):
        errors = []

        if not self.states:
            errors.append("DFA must have at least one state")

        if self.start_state is None:
            errors.append("DFA must have a start state")
        elif self.start_state not in self.states:
            errors.append("Start state must be one of the defined states")

        if not self.alphabet:
            errors.append("DFA must have at least one symbol in alphabet")

        return len(errors) == 0, errors

    def expand_wildcard_string(self, input_string):
        """Expand wildcard string to generate test cases"""
        if '*' not in input_string and '?' not in input_string:
            return [input_string]

        # Generate test cases for wildcards
        test_cases = []

        # Simple expansion - replace * with 0-3 repetitions, ? with each alphabet symbol
        def expand_recursive(s, index=0):
            if index >= len(s):
                return [s]

            results = []
            if s[index] == '*':
                # Get the previous character for repetition
                if index > 0:
                    prev_char = s[index - 1]
                    # Replace * with 0, 1, 2, 3 repetitions of previous character
                    for rep in range(4):
                        new_s = s[:index] + prev_char * rep + s[index + 1:]
                        results.extend(expand_recursive(new_s, index + rep))
                else:
                    # If * is at the beginning, just remove it
                    new_s = s[index + 1:]
                    results.extend(expand_recursive(new_s, index))
            elif s[index] == '?':
                # Replace ? with each alphabet symbol
                for symbol in sorted(self.alphabet):
                    new_s = s[:index] + symbol + s[index + 1:]
                    results.extend(expand_recursive(new_s, index + 1))
            else:
                results.extend(expand_recursive(s, index + 1))

            return results

        expanded = expand_recursive(input_string)
        # Limit to first 10 test cases to avoid overwhelming
        return list(set(expanded))[:10]

    def simulate(self, input_string):
        if not self.is_valid()[0]:
            return False, [], "DFA is not valid"

        current_state = self.start_state
        path = [current_state]
        step_details = []

        for i, symbol in enumerate(input_string):
            if symbol not in self.alphabet:
                return False, path, f"Symbol '{symbol}' not in alphabet"

            # Check if transition exists, if not, go to implicit reject state
            if (current_state, symbol) in self.transitions:
                next_state = self.transitions[(current_state, symbol)]
                current_state = next_state
                path.append(current_state)
                step_details.append({
                    'step': i + 1,
                    'symbol': symbol,
                    'from_state': path[i],
                    'to_state': current_state,
                    'status': 'valid_transition'
                })
            else:
                # Implicit reject - transition to non-existent reject state
                step_details.append({
                    'step': i + 1,
                    'symbol': symbol,
                    'from_state': current_state,
                    'to_state': 'REJECT',
                    'status': 'no_transition'
                })
                return False, path, f"No transition from {current_state} on '{symbol}' - String rejected"

        accepted = current_state in self.accept_states
        return accepted, path, "String accepted" if accepted else "String rejected", step_details

    def simulate_step_by_step(self, input_string):
        """Generator for step-by-step simulation"""
        if not self.is_valid()[0]:
            yield False, [], "DFA is not valid", []
            return

        current_state = self.start_state
        path = [current_state]

        # Initial state
        yield None, path, f"Starting at state: {current_state}", []

        for i, symbol in enumerate(input_string):
            if symbol not in self.alphabet:
                yield False, path, f"Symbol '{symbol}' not in alphabet", []
                return

            if (current_state, symbol) in self.transitions:
                next_state = self.transitions[(current_state, symbol)]
                current_state = next_state
                path.append(current_state)
                yield None, path, f"Read '{symbol}' → transition to {current_state}", []
            else:
                yield False, path, f"No transition from {current_state} on '{symbol}' - REJECTED", []
                return

        # Final result
        accepted = current_state in self.accept_states
        yield accepted, path, "ACCEPTED" if accepted else "REJECTED", []


def create_sample_dfas():
    """Create sample DFAs for demonstration"""
    sample_dfas = {}

    # Sample 1: Binary strings ending with '01'
    dfa1 = DFA("Binary strings ending with '01'")
    dfa1.alphabet = {'0', '1'}
    dfa1.add_state('q0', is_start=True)
    dfa1.add_state('q1')
    dfa1.add_state('q2', is_accept=True)
    dfa1.transitions = {
        ('q0', '0'): 'q1',
        ('q0', '1'): 'q0',
        ('q1', '0'): 'q1',
        ('q1', '1'): 'q2',
        ('q2', '0'): 'q1',
        ('q2', '1'): 'q0'
    }
    sample_dfas['sample1'] = dfa1

    # Sample 2: Even number of 'a's
    dfa2 = DFA("Even number of 'a's")
    dfa2.alphabet = {'a', 'b'}
    dfa2.add_state('q0', is_start=True, is_accept=True)
    dfa2.add_state('q1')
    dfa2.transitions = {
        ('q0', 'a'): 'q1',
        ('q0', 'b'): 'q0',
        ('q1', 'a'): 'q0',
        ('q1', 'b'): 'q1'
    }
    sample_dfas['sample2'] = dfa2

    # Sample 3: Contains substring 'abc'
    dfa3 = DFA("Contains substring 'abc'")
    dfa3.alphabet = {'a', 'b', 'c'}
    dfa3.add_state('q0', is_start=True)
    dfa3.add_state('q1')
    dfa3.add_state('q2')
    dfa3.add_state('q3', is_accept=True)
    dfa3.transitions = {
        ('q0', 'a'): 'q1',
        ('q0', 'b'): 'q0',
        ('q0', 'c'): 'q0',
        ('q1', 'a'): 'q1',
        ('q1', 'b'): 'q2',
        ('q1', 'c'): 'q0',
        ('q2', 'a'): 'q1',
        ('q2', 'b'): 'q0',
        ('q2', 'c'): 'q3',
        ('q3', 'a'): 'q3',
        ('q3', 'b'): 'q3',
        ('q3', 'c'): 'q3'
    }
    sample_dfas['sample3'] = dfa3

    return sample_dfas


def validate_state_name(state_name):
    """Validate state name input"""
    if not state_name:
        return False, "State name cannot be empty"
    if not re.match(r'^[a-zA-Z0-9_]+$', state_name):
        return False, "State name can only contain letters, numbers, and underscores"
    return True, ""


def validate_symbol(symbol):
    """Validate alphabet symbol input"""
    if not symbol:
        return False, "Symbol cannot be empty"
    if len(symbol) != 1:
        return False, "Symbol must be a single character"
    if not re.match(r'^[a-zA-Z0-9]$', symbol):
        return False, "Symbol must be alphanumeric"
    return True, ""


def draw_dfa_graph(dfa, highlight_path=None, current_state_highlight=None):
    """Create a visualization of the DFA"""
    if not dfa.states:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for state in dfa.states:
        G.add_node(state)

    # Add edges with labels
    edge_labels = {}
    edges_dict = defaultdict(list)

    for (from_state, symbol), to_state in dfa.transitions.items():
        edges_dict[(from_state, to_state)].append(symbol)

    for (from_state, to_state), symbols in edges_dict.items():
        G.add_edge(from_state, to_state)
        edge_labels[(from_state, to_state)] = ','.join(sorted(symbols))

    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if current_state_highlight and node == current_state_highlight:
            node_colors.append('yellow')  # Current state in simulation
        elif highlight_path and node in highlight_path:
            if node == dfa.start_state:
                node_colors.append('lightgreen')
            elif node in dfa.accept_states:
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightyellow')
        else:
            if node == dfa.start_state and node in dfa.accept_states:
                node_colors.append('gold')
            elif node == dfa.start_state:
                node_colors.append('lightgreen')
            elif node in dfa.accept_states:
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)

    # Draw edges
    edge_colors = []
    for edge in G.edges():
        if highlight_path and len(highlight_path) > 1:
            path_edges = [(highlight_path[i], highlight_path[i + 1]) for i in range(len(highlight_path) - 1)]
            if edge in path_edges:
                edge_colors.append('red')
            else:
                edge_colors.append('gray')
        else:
            edge_colors.append('black')

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True,
                           arrowsize=20, arrowstyle='->', width=2, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax)

    # Add legend
    legend_elements = [
        patches.Patch(color='lightgreen', label='Start State'),
        patches.Patch(color='lightcoral', label='Accept State'),
        patches.Patch(color='gold', label='Start & Accept State'),
        patches.Patch(color='lightblue', label='Regular State')
    ]
    if current_state_highlight:
        legend_elements.append(patches.Patch(color='yellow', label='Current State'))

    ax.legend(handles=legend_elements, loc='upper right')

    title = f'DFA: {dfa.name}'
    if current_state_highlight:
        title += f' (Current: {current_state_highlight})'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    return fig


# Main application
st.title("DFA Simulator")

# Sidebar
with st.sidebar:
    st.header("DFA Creation")

    # DFA Name
    dfa_name = st.text_input("DFA Name", placeholder="Enter DFA name")

    if dfa_name and dfa_name not in st.session_state.dfas:
        st.session_state.dfas[dfa_name] = DFA(dfa_name)
        st.session_state.current_dfa = dfa_name

    # Select current DFA
    if st.session_state.dfas:
        current_dfa_name = st.selectbox("Selected DFA", list(st.session_state.dfas.keys()))
        st.session_state.current_dfa = current_dfa_name

    if st.session_state.current_dfa:
        with st.expander('Set up DFA', width='stretch'):
            current_dfa = st.session_state.dfas[st.session_state.current_dfa]

            # Add States
            with st.popover("Add States", use_container_width=True):
                st.subheader("Add State")
                new_state = st.text_input("State Name", key="new_state")
                is_start = st.checkbox("Start State", key="is_start_state")
                is_accept = st.checkbox("Accept State", key="is_accept_state")

                if st.button("Add State", key="add_state_btn"):
                    if new_state:
                        is_valid, error_msg = validate_state_name(new_state)
                        if is_valid:
                            if new_state not in current_dfa.states:
                                current_dfa.add_state(new_state, is_start, is_accept)
                                st.success(f"State '{new_state}' added successfully")
                                st.rerun()
                            else:
                                st.error("State already exists")
                        else:
                            st.error(error_msg)
                    else:
                        st.error("Please enter a state name")

            # Add Alphabet
            with st.popover("Add Alphabet", use_container_width=True):
                st.subheader("Add Symbol")
                new_symbol = st.text_input("Symbol", key="new_symbol")

                if st.button("Add Symbol", key="add_symbol_btn"):
                    if new_symbol:
                        is_valid, error_msg = validate_symbol(new_symbol)
                        if is_valid:
                            if new_symbol not in current_dfa.alphabet:
                                current_dfa.alphabet.add(new_symbol)
                                st.success(f"Symbol '{new_symbol}' added successfully")
                                st.rerun()
                            else:
                                st.error("Symbol already exists")
                        else:
                            st.error(error_msg)
                    else:
                        st.error("Please enter a symbol")

            # Add Transitions
            with st.popover("Add Transitions", use_container_width=True):
                st.subheader("Add Transition")
                if current_dfa.states and current_dfa.alphabet:
                    from_state = st.selectbox("From State", list(current_dfa.states), key="from_state")
                    symbol = st.selectbox("On Symbol", list(current_dfa.alphabet), key="on_symbol")
                    to_state = st.selectbox("To State", list(current_dfa.states), key="to_state")

                    if st.button("Add Transition", key="add_transition_btn"):
                        if current_dfa.add_transition(from_state, symbol, to_state):
                            st.success(f"Transition ({from_state}, {symbol}) -> {to_state} added")
                            st.rerun()
                        else:
                            st.error("Failed to add transition")
                else:
                    st.warning("Please add states and alphabet symbols first")

            st.info("Note: Unspecified transitions automatically lead to rejection")

    st.divider()

    # Example DFAs
    st.header("Examples")
    sample_dfas = create_sample_dfas()


    if st.button("Binary '01'", use_container_width=True):
        st.session_state.dfas['sample1'] = sample_dfas['sample1']
        st.session_state.current_dfa = 'sample1'
        st.rerun()

    if st.button("Even 'a's", use_container_width=True):
        st.session_state.dfas['sample2'] = sample_dfas['sample2']
        st.session_state.current_dfa = 'sample2'
        st.rerun()

    if st.button("Contains 'abc'", use_container_width=True):
        st.session_state.dfas['sample3'] = sample_dfas['sample3']
        st.session_state.current_dfa = 'sample3'
        st.rerun()

    st.divider()

    # DFA Management
    st.header("DFA Management")
    if st.session_state.current_dfa and st.button("Delete Current DFA", type="secondary"):
        del st.session_state.dfas[st.session_state.current_dfa]
        st.session_state.current_dfa = None if not st.session_state.dfas else list(st.session_state.dfas.keys())[0]
        st.rerun()

# Main content
if st.session_state.current_dfa:
    current_dfa = st.session_state.dfas[st.session_state.current_dfa]

    tab1, tab2 = st.tabs(["DFA Overview", "Simulation"])

    with tab1:
        st.subheader(f"Overview: {current_dfa.name}")

        # Validation
        is_valid, errors = current_dfa.is_valid()

        if not is_valid:
            st.error("DFA Configuration Errors:")
            for error in errors:
                st.error(f"- {error}")
        else:
            st.success("DFA is correctly configured")

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("States", len(current_dfa.states))
        with col2:
            st.metric("Alphabet Size", len(current_dfa.alphabet))
        with col3:
            st.metric("Transitions", len(current_dfa.transitions))
        with col4:
            st.metric("Accept States", len(current_dfa.accept_states))

        # Current configuration
        if current_dfa.states:
            st.subheader("Configuration Details")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**States:**", ", ".join(sorted(current_dfa.states)))
                st.write("**Alphabet:**", ", ".join(sorted(current_dfa.alphabet)))

            with col2:
                st.write("**Start State:**", current_dfa.start_state or "Not set")
                st.write("**Accept States:**", ", ".join(sorted(current_dfa.accept_states)) or "None")

        # Transition table
        if current_dfa.transitions and current_dfa.states and current_dfa.alphabet:
            st.subheader("Transition Table")

            # Create transition table
            states_list = sorted(current_dfa.states)
            symbols_list = sorted(current_dfa.alphabet)

            table_data = []
            for state in states_list:
                row = [state]
                for symbol in symbols_list:
                    next_state = current_dfa.transitions.get((state, symbol), "REJECT")
                    row.append(next_state)
                table_data.append(row)

            df = pd.DataFrame(table_data, columns=["State"] + symbols_list)
            st.dataframe(df, use_container_width=True)
            st.caption("Note: REJECT indicates unspecified transitions that lead to automatic rejection")

        # Visualization
        if is_valid:
            st.subheader("DFA Visualization")
            fig = draw_dfa_graph(current_dfa)
            if fig:
                st.pyplot(fig)

    with tab2:
        st.subheader("DFA Simulation")

        if not current_dfa.is_valid()[0]:
            st.error("Please fix the DFA configuration errors before simulation")
        else:
            # Show DFA graph at the top
            colL, colR = st.columns(2)
            with colR:
                fig = draw_dfa_graph(current_dfa)
                graph_placeholder = st.empty()
                if fig:
                    graph_placeholder.pyplot(fig)

            # Input string
            with colL:
                test_string = st.text_input("Enter test string (supports * and ? wildcards)",
                                            placeholder="Enter string to test (e.g., ab*c, a?b)")
                # Wildcard help
                st.caption("Wildcards: * repeats previous character 0-3 times, ? matches any alphabet symbol")
                auto_sim = st.checkbox("Auto-simulate", value=st.session_state.auto_simulate)
                st.session_state.auto_simulate = auto_sim

                col1, sep, col2 = st.columns([1,2,1])
                with col1:
                    simulate_btn = st.button("Simulate", type="primary")

                with col2:
                    if st.button("Reset Simulation"):
                        st.session_state.current_step = 0
                        st.rerun()


                if simulate_btn and test_string is not None:
                    # Handle wildcards
                    if '*' in test_string or '?' in test_string:
                        st.subheader("Wildcard Expansion")
                        expanded_strings = current_dfa.expand_wildcard_string(test_string)
                        st.write(f"Testing {len(expanded_strings)} expanded strings:")

                        for expanded_string in expanded_strings:
                            st.write(f"- {expanded_string}")

                            # Validate each expanded string
                            invalid_symbols = [c for c in expanded_string if c not in current_dfa.alphabet]
                            if invalid_symbols:
                                st.error(f"Invalid symbols in '{expanded_string}': {', '.join(set(invalid_symbols))}")
                            else:
                                result = current_dfa.simulate(expanded_string)
                                accepted, path, message = result[0], result[1], result[2]

                                if accepted:
                                    st.success(f"'{expanded_string}' → ACCEPTED")
                                else:
                                    st.error(f"'{expanded_string}' → REJECTED")
                    else:
                        # Regular simulation
                        invalid_symbols = [c for c in test_string if c not in current_dfa.alphabet]
                        if invalid_symbols:
                            st.error(f"Invalid symbols in input: {', '.join(set(invalid_symbols))}")
                        else:
                            if st.session_state.auto_simulate:
                                # Auto-simulation with step-by-step visualization
                                st.subheader("Auto-Simulation")

                                simulation_container = st.container()
                                step_container = st.container()

                                simulator = current_dfa.simulate_step_by_step(test_string)

                                for step_result in simulator:
                                    accepted, path, message, step_details = step_result

                                    with step_container:
                                        st.write(f"**Step:** {message}")

                                    # Update graph with current state
                                    if path:
                                        current_state = path[-1]
                                        fig = draw_dfa_graph(current_dfa, highlight_path=path,
                                                             current_state_highlight=current_state)
                                        if fig:
                                            graph_placeholder.pyplot(fig)

                                    # Add delay for visualization
                                    time.sleep(1)

                                    if accepted is not None:  # Final result
                                        if accepted:
                                            st.success("FINAL RESULT: String ACCEPTED")
                                        else:
                                            st.error("FINAL RESULT: String REJECTED")
                                        break

                            else:
                                # Standard simulation
                                result = current_dfa.simulate(test_string)
                                accepted, path, message = result[0], result[1], result[2]
                                step_details = result[3] if len(result) > 3 else []

                                # Show result
                                if accepted:
                                    st.success(f"String '{test_string}' is ACCEPTED")
                                else:
                                    st.error(f"String '{test_string}' is REJECTED")

                                # Show simulation path
                                st.write("**Simulation Path:**")
                                if path:
                                    path_str = " → ".join(path)
                                    st.write(path_str)

                                # Show step-by-step
                                if step_details:
                                    with st.popover("**Step-by-step Execution:**", use_container_width=True):
                                        for step in step_details:
                                            if step['status'] == 'valid_transition':
                                                st.write(f"Step {step['step']}: Read '{step['symbol']}', "
                                                         f"from {step['from_state']} → {step['to_state']}")
                                            else:
                                                st.write(f"Step {step['step']}: Read '{step['symbol']}', "
                                                         f"no transition from {step['from_state']} → REJECT")
                                elif test_string:
                                    for i, (symbol, state) in enumerate(
                                            zip(test_string, path[1:] if len(path) > 1 else [])):
                                        st.write(f"Step {i + 1}: Read '{symbol}', transition to state '{state}'")

                                # Update visualization with highlighted path
                                fig = draw_dfa_graph(current_dfa, highlight_path=path)
                                if fig:
                                    graph_placeholder.pyplot(fig)

                                # Add to history
                                st.session_state.simulation_history.append({
                                    'dfa': current_dfa.name,
                                    'input': test_string,
                                    'result': 'ACCEPTED' if accepted else 'REJECTED',
                                    'path': " → ".join(path) if path else "N/A"
                                })

            st.divider()

            # Simulation history
            if st.session_state.simulation_history:
                st.subheader("Simulation History")
                history_df = pd.DataFrame(st.session_state.simulation_history)
                st.dataframe(history_df, use_container_width=True)

                if st.button("Clear History"):
                    st.session_state.simulation_history = []
                    st.rerun()

else:
    st.info("Please create or select a DFA to get started")
    st.write("Use the sidebar to:")
    st.write("- Create a new DFA by entering a name")
    st.write("- Load one of the example DFAs")
    st.write("- Add states, alphabet symbols, and transitions")
