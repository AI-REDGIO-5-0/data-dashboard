import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
import scipy.stats
import json
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from rdflib import Graph, Namespace, URIRef
from rdflib.plugins.sparql import prepareQuery
from pyvis.network import Network
from streamlit.components.v1 import html
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import urllib.parse
import uuid

# Function to create a unique key
def generate_unique_key(prefix=''):
    return f'{prefix}_{uuid.uuid4()}'

# Function to create the RDF graph representing the knowledge graph
def create_knowledge_graph(correlations):
    """Create an RDF graph based on correlations between data attributes."""
    g = Graph()
    ns = Namespace("https://www.ai-redgio.eu/")
    for pair, pearson_corr, spearman_corr, euclidean_sim in correlations:
        attr1_uri = URIRef(ns[pair[0]])
        attr2_uri = URIRef(ns[pair[1]])
        # Add Pearson correlation relationship
        if pearson_corr > 0.5:  # Assuming a threshold of 0.5 for significant correlation
            g.add((attr1_uri, URIRef(ns["hasSignificantPearsonCorrelationWith"]), attr2_uri))
        # Add Spearman correlation relationship
        if spearman_corr > 0.5:
            g.add((attr1_uri, URIRef(ns["hasSignificantSpearmanCorrelationWith"]), attr2_uri))
        # Add Euclidean similarity relationship
        if euclidean_sim > 0.5:  # Assuming a threshold of 0.5 for significant similarity
            g.add((attr1_uri, URIRef(ns["hasHighEuclideanSimilarityWith"]), attr2_uri))
    return g
    
def apply_rules_to_graph(g, rules_text, ns):
    """Parse and apply IF-THEN rules to the RDF graph."""
    rules = rules_text.split('\n')  # Split rules by newline
    for rule in rules:
        rule = rule.strip()
        if rule and "IF" in rule and "THEN" in rule:
            conditions_part, result_part = rule.split("THEN")
            _, conditions_str = conditions_part.split("IF")
            conditions = [cond.strip() for cond in conditions_str.split("AND")]
            result_predicate = result_part.strip()

            for s in set(g.subjects()):  # Iterate over subjects in the graph
                add_relation = True
                for condition in conditions:
                    if ">" in condition:
                        predicate, condition_value = condition.split(">")
                        predicate = predicate.strip()
                        condition_value = float(condition_value.strip())

                        found = False
                        for _, _, value in g.triples((s, ns[predicate], None)):
                            if float(value) > condition_value:
                                found = True
                                break
                        if not found:
                            add_relation = False
                            break
                    else:
                        predicate = condition.strip()
                        if not (s, ns[predicate], None) in g:
                            add_relation = False
                            break

                if add_relation:
                    # Assuming the object to be added is `Literal(True)` or `RDF.type`
                    g.add((s, ns[result_predicate], Literal(True)))  # Adjust according to actual use case


# Function to visualize the RDF graph as a network
def visualize_knowledge_graph(g):
    """Visualize the RDF graph as a network with enhanced aesthetics."""
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")

    # Node and Edge Customization
    for subj, pred, obj in g:
        subj_name = str(subj).split('/')[-1]
        obj_name = str(obj).split('/')[-1]
        pred_name = str(pred).split('/')[-1]

        net.add_node(subj_name, label=subj_name, title=subj_name, color="#f0a30a", size=15, font={"color": "white", "size": 12})
        net.add_node(obj_name, label=obj_name, title=obj_name, color="#7DCE13", size=10, font={"color": "white", "size": 10})
        net.add_edge(subj_name, obj_name, title=pred_name, color="white", width=2, arrowStrikethrough=True)

    # Layout Options
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidthSelected": 2
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": {
          "type": "continuous"
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95
        },
        "minVelocity": 0.75
      }
    }
    """)

    net.show("knowledge_graph.html")

    # Display the network visualization in the Streamlit app
    html_file = open("knowledge_graph.html", "r", encoding="utf-8").read()
    html(html_file, width=800, height=750)

    
    # SPARQL Query Interface
    st.write("### SPARQL Query Interface")
    query_text = st.text_input('Enter a SPARQL query', value='SELECT ?subject WHERE { ?subject <https://www.ai-redgio.eu/_pearson> ?object . }', key=generate_unique_key('sparql_query'))

    if query_text:
        # Prepare the SPARQL query
        ns = {"example": Namespace("https://www.ai-redgio.eu/")}
        query = prepareQuery(query_text, initNs=ns)

        # Execute the SPARQL query against the RDF graph
        results = g.query(query)

        # Process and Display Results
        processed_results = process_sparql_results(results)
        st.write(processed_results)
        
    # Inside your Streamlit application, add an interface for rule input
    st.write("### Inference")
    rule_text = st.text_area("Enter your IF-THEN rules", value="IF _pearson AND _spearman THEN _strongCorrelation", height=100)
    execute_rules = st.button("Execute Rules")
    
    if execute_rules:
        ns = Namespace("https://www.ai-redgio.eu/")
        apply_rules_to_graph(g, rule_text, ns)
        visualize_knowledge_graph(g)
    


def process_sparql_results(results):
    """Process SPARQL query results into a readable format."""
    processed_results = []
    for row in results:
        processed_row = {str(k): str(v) for k, v in row.asdict().items()}
        processed_results.append(processed_row)
    return processed_results


# Function to calculate Euclidean similarity between two vectors
def euclidean_similarity(vector1, vector2):
    """Calculate Euclidean similarity between two vectors."""
    distance = np.linalg.norm(vector1 - vector2)
    similarity = 1 / (1 + distance)
    return similarity

# Function to preprocess the dataset
def preprocess_dataset(file_path, impute_method='ffill', numeric_method='coerce', categorical_method='label'):
    """Preprocess the dataset by reading from a path, handling missing values,
    converting data types, and encoding categorical variables."""
    # Read CSV file into DataFrame
    df = pd.read_csv(file_path)

    # Handle data types and missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = handle_data_types(df[col], numeric_method, categorical_method)
    df = impute_missing_values(df, impute_method)
    return df

# Function to handle data types for a column
def handle_data_types(column, numeric_method, categorical_method):
    """Convert or encode column data based on specified methods."""
    if numeric_method == 'coerce':
        return pd.to_numeric(column, errors='coerce')
    elif categorical_method == 'label':
        return LabelEncoder().fit_transform(column)
    return column

# Function to impute missing values in the DataFrame
def impute_missing_values(df, method):
    """Impute missing values in the DataFrame based on the specified method."""
    if method == 'ffill':
        return df.fillna(method='ffill')
    if method in ['mean', 'median']:
        return df.fillna(df.agg(method))
    return df

# Function to handle file upload or URL input
def handle_file_input():
    """Handle file input either by direct upload or via URL and return the file content."""
    # Use st.query_params instead of st.experimental_get_query_params
    file_url = st.query_params.get('file_url')
    if file_url:
        return fetch_file_from_url(file_url[0])
    return st.file_uploader('Choose a CSV file', type=['csv'], key='csv_file_uploader')


# Function to fetch file from a URL
def fetch_file_from_url(url):
    """Fetch a file from the specified URL."""
    if url.startswith(('http://', 'https://')):
        try:
            return requests.get(url).content
        except Exception as e:
            st.error(f'Error reading the file from URL: {e}')
    elif os.path.isfile(url):
        return url
    st.error(f'Invalid file path or URL: {url}')
    return None

# Function to handle file upload and preprocess it
def handle_file_upload():
    """Allow users to upload a CSV file and preprocess the data."""
    uploaded_file = st.file_uploader('Upload a CSV file for analysis', type=['csv'], key='file_uploader')
    if uploaded_file is not None:
        try:
            # Load the file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            return df
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    else:
        st.info("Please upload a CSV file.")
    return None

# Main function to run the Streamlit app
def main():
    """Main function to run the Streamlit app for data analysis and visualization."""
    
    # Display logo (if exists)
    if os.path.exists("logo.png"):
        st.image("logo.png", width=400)
    else:
        st.write("Logo not found.")

    # Title and Information
    st.title('Data Analysis Dashboard')

    # Handle file upload
    df = handle_file_upload()

    if df is not None:
        # Perform data analysis if a valid DataFrame is uploaded
        perform_data_analysis(df)
    else:
        st.warning("Awaiting file upload for further analysis.")




# --- Helper Functions (Assumptions) ---

def process_uploaded_file(file_path):
    """Function to process the uploaded file (e.g., read CSV/Excel)."""
    try:
        # Example: Assuming CSV for demonstration
        df = pd.read_csv(file_path)
        return df
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
        return None
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        return None

# Function to perform data analysis on the DataFrame
# Integrate the knowledge graph creation and visualization into the main analysis flow
# Extended perform_data_analysis function with threshold sliders
def perform_data_analysis(df):
    """Perform data analysis including correlation analysis, visualization, and knowledge graph creation with user-defined thresholds."""
    selected_columns = st.multiselect('Select columns to compare', df.columns)
    if selected_columns:
        selected_pairs = select_column_pairs(selected_columns)
        
        # User-defined thresholds for correlations and similarities
        pearson_threshold = st.sidebar.slider('Pearson correlation threshold', 0.0, 1.0, 0.5, 0.01)
        spearman_threshold = st.sidebar.slider('Spearman correlation threshold', 0.0, 1.0, 0.5, 0.01)
        euclidean_threshold = st.sidebar.slider('Euclidean similarity threshold', 0.0, 1.0, 0.5, 0.01)

        correlations = calculate_correlations(df, selected_pairs)
        visualize_correlations(df, correlations)

        # Pass user-defined thresholds to the knowledge graph creation function
        kg = create_knowledge_graph(correlations, pearson_threshold, spearman_threshold, euclidean_threshold)
        visualize_knowledge_graph(kg)

# Modified create_knowledge_graph function to accept thresholds
def create_knowledge_graph(correlations, pearson_threshold, spearman_threshold, euclidean_threshold):
    """Create an RDF graph based on correlations between data attributes with user-defined thresholds."""
    g = Graph()
    ns = Namespace("https://www.ai-redgio.eu/")
    for pair, pearson_corr, spearman_corr, euclidean_sim in correlations:
        attr1_uri = URIRef(ns[pair[0]])
        attr2_uri = URIRef(ns[pair[1]])
        # Add Pearson correlation relationship based on user-defined threshold
        if pearson_corr > pearson_threshold:
            g.add((attr1_uri, URIRef(ns["_pearson"]), attr2_uri))
        # Add Spearman correlation relationship based on user-defined threshold
        if spearman_corr > spearman_threshold:
            g.add((attr1_uri, URIRef(ns["_spearman"]), attr2_uri))
        # Add Euclidean similarity relationship based on user-defined threshold
        if euclidean_sim > euclidean_threshold:
            g.add((attr1_uri, URIRef(ns["_euclidean"]), attr2_uri))
    return g
    
    
# Function to select column pairs for comparison
def select_column_pairs(columns):
    """Let users select pairs of columns for comparison."""
    selected_pairs = []
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i + 1:]):
            if st.sidebar.checkbox(f'{col1} vs {col2}', value=True):
                selected_pairs.append((col1, col2))
    return selected_pairs

# Function to display correlations and visualizations for selected column pairs
def display_correlations(df, selected_pairs):
    """Display correlations and visualizations for selected column pairs."""
    if selected_pairs:
        correlations = calculate_correlations(df, selected_pairs)
        visualize_correlations(df, correlations)

# Function to calculate correlations for selected column pairs
def calculate_correlations(df, selected_pairs):
    """Calculate Pearson and Spearman correlations and Euclidean similarity for selected column pairs."""
    correlations = []
    for pair in selected_pairs:
        pearson_corr = np.corrcoef(df[pair[0]], df[pair[1]])[0, 1]
        spearman_corr, _ = scipy.stats.spearmanr(df[pair[0]], df[pair[1]])
        euclidean_sim = euclidean_similarity(df[pair[0]].fillna(0), df[pair[1]].fillna(0))
        correlations.append((pair, pearson_corr, spearman_corr, euclidean_sim))
    return correlations

# Function to visualize correlations using Plotly
def visualize_correlations(df, correlations):
    """Visualize correlations using Plotly subplots."""
    fig = create_correlation_subplots(df, correlations)
    st.plotly_chart(fig, use_container_width=True)

# Function to create Plotly subplots for correlations
def create_correlation_subplots(df, correlations):
    """Create Plotly subplots for each correlation pair."""
    fig = sp.make_subplots(rows=len(correlations), cols=1, subplot_titles=[f'{pair[0]} vs {pair[1]}' for pair, _, _, _ in correlations])
    for i, (pair, pearson_corr, spearman_corr, euclidean_sim) in enumerate(correlations, start=1):
        trace1 = go.Scatter(x=df.index, y=df[pair[0]], name=pair[0])
        trace2 = go.Scatter(x=df.index, y=df[pair[1]], name=pair[1])
        fig.add_trace(trace1, row=i, col=1)
        fig.add_trace(trace2, row=i, col=1)
        fig.update_xaxes(title_text="Index", row=i, col=1)
        fig.update_yaxes(title_text="Value", row=i, col=1)
    fig.update_layout(height=300 * len(correlations), title_text="Correlation Analysis")
    return fig

if __name__ == '__main__':
    main()
