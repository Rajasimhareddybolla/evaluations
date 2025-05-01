# Fix for PyTorch/Streamlit compatibility issue - Must be at the very top of the file
import os
# Configure Streamlit to exclude torch from module watching
os.environ["STREAMLIT_WATCH_EXCLUDES"] = "torch,torchvision,torchaudio,torch._classes"
# Also exclude specific problematic modules
os.environ["STREAMLIT_SERVER_WATCH_EXCLUDES"] = "torch,torchvision,torchaudio,torch._classes"

import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import altair as alt
from utils import util, Gemini, Accuracy, c_c_prompt, saftey

# Set page config
st.set_page_config(
    page_title="Text Evaluation Dashboard",

    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize utility classes
@st.cache_resource
def load_resources():
    return util(), Gemini()

# Make sure torch is imported after the environment variables are set
# This prevents the error from occurring
with st.spinner("Loading models..."):
    utils_obj, gemini_obj = load_resources()

# Session state initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = []

# Sidebar
st.sidebar.title("Evaluation Dashboard")
st.sidebar.info(
    "This app evaluates text using multiple metrics including accuracy, "
    "completeness, clarity, relevance, coherence, readability, and safety."
)

tab_options = ["New Evaluation", "History", "Metrics Explanation"]
selected_tab = st.sidebar.radio("Navigation", tab_options)

# Functions for evaluation
@st.cache_data
def evaluate_accuracy(input_text, output_text, context=""):
    """Evaluate accuracy using Gemini API"""
    prompt_template = Accuracy[0].replace("{{input}}", input_text).replace("{{output}}", output_text).replace("{{context}}", context)
    try:
        response = gemini_obj.generate(prompt_template , Accuracy[1])
        result = json.loads(response)
        return {
            "accuracy": result.get("accuracy", "Error in processing"),
            "accuracy_exp": result.get("accuracy_exp", "Explanation not available")
        }
    except Exception as e:
        st.error(f"Error in accuracy evaluation: {str(e)}")
        return {"accuracy": "Error in processing", "accuracy_exp": "Error occurred during evaluation"}

@st.cache_data
def evaluate_cc_metrics(input_text, output_text):
    """Evaluate completeness, clarity, and relevance using Gemini API"""
    prompt_template = c_c_prompt[0].replace("{{input}}", input_text).replace("{{output}}", output_text)
    try:
        # Use an empty dictionary (or a proper config) for the second argument
        response = gemini_obj.generate(prompt_template, {
  "type": "object",
  "properties": {
    "completeness": {
      "type": "number"
    },
    "clarity": {
      "type": "number"
    },
    "relevance": {
      "type": "number"
    }
  },
  "required": [
    "completeness",
    "clarity",
    "relevance"
  ]
})
        result = json.loads(response)
        return {
            "completeness": result.get("completeness", 0),
            "clarity": result.get("clarity", 0),
            "relevance": result.get("relevance", 0)
        }
    except Exception as e:
        st.error(f"Error in completeness/clarity evaluation: {str(e)}")
        return {"completeness": 0, "clarity": 0, "relevance": 0}

@st.cache_data
def evaluate_safety(output_text):
    """Evaluate safety/toxicity using Gemini API"""
    prompt_template = saftey[0].replace("{{output}}", output_text)
    try:
        response = gemini_obj.generate(prompt_template  , saftey[1])
        result = json.loads(response)
        return result.get("safety_score", 0)
    except Exception as e:
        st.error(f"Error in safety evaluation: {str(e)}")
        return 0

@st.cache_data
def evaluate_coherence(text):
    """Evaluate coherence using utility function"""
    return utils_obj.evaluate_coherence(text)

@st.cache_data
def evaluate_readability(text):
    """Evaluate readability using utility functions"""
    dale_chall = utils_obj.dale_chall(text)
    gunning_fog = utils_obj.gunning_fog_formula(text)
    flesch = utils_obj.flesch_reading_ease(text)
    
    # Interpret scores
    dale_chall_interpretation = interpret_dale_chall(dale_chall)
    gunning_fog_interpretation = interpret_gunning_fog(gunning_fog)
    flesch_interpretation = interpret_flesch(flesch)
    
    return {
        "dale_chall": {
            "score": dale_chall,
            "interpretation": dale_chall_interpretation
        },
        "gunning_fog": {
            "score": gunning_fog,
            "interpretation": gunning_fog_interpretation
        },
        "flesch": {
            "score": flesch,
            "interpretation": flesch_interpretation
        }
    }

# Helper functions for readability interpretation
def interpret_dale_chall(score):
    if score <= 4.9:
        return "Very easy (Grade 4 and below)"
    elif score <= 5.9:
        return "Easy (Grades 5-6)"
    elif score <= 6.9:
        return "Fairly easy (Grades 7-8)"
    elif score <= 7.9:
        return "Standard (Grades 9-10)"
    elif score <= 8.9:
        return "Fairly difficult (Grades 11-12)"
    elif score <= 9.9:
        return "Difficult (College)"
    else:
        return "Very difficult (College graduate)"

def interpret_gunning_fog(score):
    if score < 6:
        return "Very easy to read"
    elif score < 8:
        return "Easy to read"
    elif score < 10:
        return "Fairly easy to read"
    elif score < 12:
        return "Standard/Plain English"
    elif score < 15:
        return "Fairly difficult to read"
    elif score < 18:
        return "Difficult to read"
    else:
        return "Very difficult to read"

def interpret_flesch(score):
    if score >= 90:
        return "Very easy to read (5th grade)"
    elif score >= 80:
        return "Easy to read (6th grade)"
    elif score >= 70:
        return "Fairly easy to read (7th grade)"
    elif score >= 60:
        return "Plain English (8-9th grade)"
    elif score >= 50:
        return "Fairly difficult (10-12th grade)"
    elif score >= 30:
        return "Difficult (College level)"
    else:
        return "Very difficult (College graduate level)"

# Function to run full evaluation
def run_evaluation(input_text, output_text, context=""):
    with st.spinner("Running evaluation..."):
        results = {}
        
        # Evaluate accuracy
        results["accuracy"] = evaluate_accuracy(input_text, output_text, context)
        
        # Evaluate completeness, clarity, relevance
        cc_metrics = evaluate_cc_metrics(input_text, output_text)
        results.update(cc_metrics)
        
        # Evaluate safety/toxicity
        results["safety_score"] = evaluate_safety(output_text)
        
        # Evaluate coherence
        coherence_results = evaluate_coherence(output_text)
        results["coherence"] = coherence_results
        
        # Evaluate readability
        readability_results = evaluate_readability(output_text)
        results["readability"] = readability_results
        
        # Add timestamp
        results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results["input_text"] = input_text
        results["output_text"] = output_text
        results["context"] = context
        
        # Save to history
        st.session_state.evaluation_results.append(results)
        
        return results

def display_radar_chart(metrics, key="radar_chart"):
    """Display radar chart for key metrics"""
    categories = ['Completeness', 'Clarity', 'Relevance', 'Coherence']
    
    # Normalize coherence to 0-10 scale if it's not already
    coherence = metrics["coherence"]["coherence_score"]
    
    values = [
        metrics["completeness"],
        metrics["clarity"],
        metrics["relevance"],
        coherence
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Evaluation Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)

def display_readability_chart(readability, key="readability_chart"):
    """Display bar chart for readability metrics"""
    # Map scores to a 0-100 scale for better visualization
    dale_chall_normalized = min(100, max(0, (15 - readability["dale_chall"]["score"]) * 10))
    gunning_fog_normalized = min(100, max(0, (25 - readability["gunning_fog"]["score"]) * 5))
    flesch_normalized = min(100, max(0, readability["flesch"]["score"]))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Dale-Chall', 'Gunning Fog', 'Flesch Reading Ease'],
        y=[dale_chall_normalized, gunning_fog_normalized, flesch_normalized],
        text=[
            f"{readability['dale_chall']['score']:.2f}",
            f"{readability['gunning_fog']['score']:.2f}",
            f"{readability['flesch']['score']:.2f}"
        ],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
    ))
    
    fig.update_layout(
        title="Readability Scores (Higher is Better)",
        xaxis_title="Metric",
        yaxis_title="Normalized Score (0-100)",
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)

# Main UI Logic
if selected_tab == "New Evaluation":
    st.title("Text Evaluation")
    
    with st.form("evaluation_form"):
        input_text = st.text_area("Input Text (Prompt/Question)", height=150)
        output_text = st.text_area("Output Text (Response to Evaluate)", height=250)
        context = st.text_area("Context (Optional)", height=150)
        
        submitted = st.form_submit_button("Run Evaluation")
    
    if submitted and output_text:
        results = run_evaluation(input_text, output_text, context)
        
        st.success("Evaluation completed!")
        
        # Display results in tabs
        tabs = st.tabs(["Summary", "Accuracy", "Completeness & Clarity", "Coherence", "Readability", "Safety", "Raw Data"])
        
        with tabs[0]:
            st.subheader("Evaluation Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                display_radar_chart(results, key="summary_radar_chart")
            
            with col2:
                display_readability_chart(results["readability"], key="summary_readability_chart")
            
            # Safety score visualization
            safety_score = results["safety_score"]
            st.subheader("Safety Score")
            
            # Create a gauge chart for safety
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=safety_score,
                title={'text': "Safety/Toxicity Risk"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "green"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': safety_score
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True, key="summary_safety_gauge")
            
            # Download button for results
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="Download Results as JSON",
                data=results_json,
                file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with tabs[1]:
            st.subheader("Accuracy Evaluation")
            st.write(f"**Score:** {results['accuracy']['accuracy']}")
            st.write(f"**Explanation:** {results['accuracy']['accuracy_exp']}")
        
        with tabs[2]:
            st.subheader("Completeness, Clarity & Relevance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Completeness", f"{results['completeness']}/10")
            with col2:
                st.metric("Clarity", f"{results['clarity']}/10")
            with col3:
                st.metric("Relevance", f"{results['relevance']}/10")
            
            # Bar chart for CC metrics
            cc_data = pd.DataFrame({
                'Metric': ['Completeness', 'Clarity', 'Relevance'],
                'Score': [results['completeness'], results['clarity'], results['relevance']]
            })
            
            chart = alt.Chart(cc_data).mark_bar().encode(
                x=alt.X('Metric', sort=None),
                y=alt.Y('Score', scale=alt.Scale(domain=[0, 10])),
                color='Metric'
            ).properties(
                width=500
            )
            
            st.altair_chart(chart, use_container_width=True)
        
        with tabs[3]:
            st.subheader("Coherence Analysis")
            coherence = results["coherence"]
            
            # Create columns for metrics and visualization
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("Coherence Score", f"{coherence['coherence_score']}/10")
                st.write(f"**Overall Quality:** {coherence.get('overall_quality', 'N/A')}")
                
                # Display paragraph count info
                if coherence.get('evaluated_paragraph_count'):
                    st.write(f"**Paragraphs Analyzed:** {coherence['evaluated_paragraph_count']}")
                
                if coherence.get('average_pairwise_coherence'):
                    st.write(f"**Average Pairwise Coherence:** {coherence['average_pairwise_coherence']:.4f}")
                
                # Display transition quality summary if available
                if coherence.get('transition_quality_summary'):
                    st.subheader("Transition Quality Distribution")
                    
                    # Create a DataFrame for the summary
                    summary_data = []
                    for quality, count in coherence['transition_quality_summary'].items():
                        summary_data.append({
                            "Quality": quality,
                            "Count": count,
                            "Percentage": f"{(count / sum(coherence['transition_quality_summary'].values()) * 100):.1f}%" if sum(coherence['transition_quality_summary'].values()) > 0 else "0%"
                        })
                    
                    # Sort by quality level (High, Medium, Low, Very Low)
                    quality_order = {"High": 1, "Medium": 2, "Low": 3, "Very Low": 4}
                    summary_data.sort(key=lambda x: quality_order.get(x["Quality"], 5))
                    
                    # Display as DataFrame
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Create pie chart for transition quality distribution
                    if sum(coherence['transition_quality_summary'].values()) > 0:
                        pie_data = pd.DataFrame({
                            'Quality': coherence['transition_quality_summary'].keys(),
                            'Count': coherence['transition_quality_summary'].values()
                        })
                        
                        # Use custom colors for different quality levels
                        colors = {
                            'High': '#2ca02c',     # Green
                            'Medium': '#1f77b4',   # Blue
                            'Low': '#ff7f0e',      # Orange
                            'Very Low': '#d62728'  # Red
                        }
                        
                        fig = px.pie(
                            pie_data, 
                            values='Count', 
                            names='Quality',
                            color='Quality',
                            color_discrete_map=colors,
                            title="Transition Quality Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show paragraph transition details if available
                if 'details' in coherence and coherence['details']:
                    # First create a table display
                    st.subheader("Paragraph Coherence Details")
                    
                    # Create a DataFrame for better display
                    detail_data = []
                    for detail in coherence['details']:
                        from_para = detail.get('from_paragraph', detail.get('between', [0, 0])[0])
                        to_para = detail.get('to_paragraph', detail.get('between', [0, 1])[1])
                        
                        detail_data.append({
                            "From": f"P{from_para+1}",
                            "To": f"P{to_para+1}",
                            "Score": detail['score'],
                            "Quality": detail['quality']
                        })
                    
                    if detail_data:
                        detail_df = pd.DataFrame(detail_data)
                        st.dataframe(detail_df, use_container_width=True)
                    
                    # Then create visual representation of coherence between paragraphs
                    scores = [detail['score'] for detail in coherence['details']]
                    
                    # Generate labels based on available fields in the details
                    if 'from_paragraph' in coherence['details'][0]:
                        labels = [f"P{detail['from_paragraph']+1}→P{detail['to_paragraph']+1}" for detail in coherence['details']]
                    else:
                        # Fallback to old format if needed
                        labels = [f"P{detail['between'][0]+1}→P{detail['between'][1]+1}" for detail in coherence['details']]
                    
                    # Color map based on score
                    colors = [
                        '#d62728' if score < 0.3 else      # Red for Very Low
                        '#ff7f0e' if score < 0.5 else      # Orange for Low
                        '#1f77b4' if score < 0.7 else      # Blue for Medium
                        '#2ca02c' for score in scores      # Green for High
                    ]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=labels,
                            y=scores,
                            marker_color=colors,
                            text=[f"{score:.4f}" for score in scores],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="Coherence Between Consecutive Paragraphs",
                        xaxis_title="Paragraph Pairs",
                        yaxis_title="Coherence Score",
                        yaxis=dict(range=[0, 1])
                    )
                    
                    # Add a horizontal line at thresholds
                    for threshold, label, color in [(0.3, "Very Low/Low", "#ff7f0e"), 
                                                    (0.5, "Low/Medium", "#1f77b4"), 
                                                    (0.7, "Medium/High", "#2ca02c")]:
                        fig.add_shape(
                            type="line",
                            x0=-0.5,
                            y0=threshold,
                            x1=len(labels)-0.5,
                            y1=threshold,
                            line=dict(color=color, width=1, dash="dash"),
                        )
                        fig.add_annotation(
                            x=len(labels)-0.5,
                            y=threshold,
                            xref="x",
                            yref="y",
                            text=label,
                            showarrow=False,
                            xanchor="right",
                            bgcolor="rgba(255,255,255,0.8)"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tabs[4]:
            st.subheader("Readability Analysis")
            readability = results["readability"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Dale-Chall Score", f"{readability['dale_chall']['score']:.2f}")
                st.write(f"**Interpretation:** {readability['dale_chall']['interpretation']}")
            
            with col2:
                st.metric("Gunning Fog Index", f"{readability['gunning_fog']['score']:.2f}")
                st.write(f"**Interpretation:** {readability['gunning_fog']['interpretation']}")
            
            with col3:
                st.metric("Flesch Reading Ease", f"{readability['flesch']['score']:.2f}")
                st.write(f"**Interpretation:** {readability['flesch']['interpretation']}")
            
            display_readability_chart(readability)
        
        with tabs[5]:
            st.subheader("Safety/Toxicity Analysis")
            safety_score = results["safety_score"]
            
            # Determine risk level
            if safety_score < 0.3:
                risk_level = "Low Risk"
                color = "green"
            elif safety_score < 0.7:
                risk_level = "Medium Risk"
                color = "orange"
            else:
                risk_level = "High Risk"
                color = "red"
            
            st.metric("Safety Score", f"{safety_score:.2f}", delta=risk_level, delta_color="normal")
            
            # Create a gauge chart for safety
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=safety_score,
                title={'text': "Safety/Toxicity Risk"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "green"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': safety_score
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True, key="safety_tab_gauge")
            
            st.write("""
            **Interpretation:**
            - 0.0 - 0.3: Minor or no safety concerns
            - 0.3 - 0.7: Moderate safety concerns
            - 0.7 - 1.0: Significant safety concerns
            """)
        
        with tabs[6]:
            st.subheader("Raw Evaluation Data")
            st.json(results)

elif selected_tab == "History":
    st.title("Evaluation History")
    
    if not st.session_state.evaluation_results:
        st.info("No evaluation history available. Run an evaluation first.")
    else:
        # Create a table of past evaluations
        history_data = []
        for i, result in enumerate(st.session_state.evaluation_results):
            history_data.append({
                "ID": i + 1,
                "Timestamp": result["timestamp"],
                "Input": result["input_text"][:50] + "..." if len(result["input_text"]) > 50 else result["input_text"],
                "Completeness": result["completeness"],
                "Clarity": result["clarity"],
                "Relevance": result["relevance"],
                "Coherence": result["coherence"]["coherence_score"],
                "Safety": result["safety_score"]
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Allow to view specific evaluation
        if history_data:
            selected_id = st.selectbox("Select evaluation to view details", options=history_df["ID"].tolist())
            
            if selected_id:
                selected_result = st.session_state.evaluation_results[selected_id - 1]
                
                st.subheader("Evaluation Details")
                
                # Display results in tabs
                tabs = st.tabs(["Summary", "Input/Output", "Metrics", "Raw Data"])
                
                with tabs[0]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_radar_chart(selected_result, key=f"history_radar_{selected_id}")
                    
                    with col2:
                        display_readability_chart(selected_result["readability"], key=f"history_readability_{selected_id}")
                
                with tabs[1]:
                    st.subheader("Input Text")
                    st.write(selected_result["input_text"])
                    
                    st.subheader("Output Text")
                    st.write(selected_result["output_text"])
                    
                    if selected_result["context"]:
                        st.subheader("Context")
                        st.write(selected_result["context"])
                
                with tabs[2]:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Completeness", f"{selected_result['completeness']}/10")
                        st.metric("Clarity", f"{selected_result['clarity']}/10")
                        st.metric("Relevance", f"{selected_result['relevance']}/10")
                    
                    with col2:
                        st.metric("Coherence", f"{selected_result['coherence']['coherence_score']}/10")
                        st.write(f"Quality: {selected_result['coherence'].get('overall_quality', 'N/A')}")
                        
                        if selected_result['coherence'].get('average_pairwise_coherence'):
                            st.write(f"Average: {selected_result['coherence']['average_pairwise_coherence']:.4f}")
                        elif selected_result['coherence'].get('average_coherence'):  # For backward compatibility
                            st.write(f"Average: {selected_result['coherence']['average_coherence']:.4f}")
                        
                        if selected_result['coherence'].get('evaluated_paragraph_count'):
                            st.write(f"Paragraphs: {selected_result['coherence']['evaluated_paragraph_count']}")
                    
                    with col3:
                        st.metric("Safety Score", f"{selected_result['safety_score']:.2f}")
                        
                        readability = selected_result["readability"]
                        st.write(f"Dale-Chall: {readability['dale_chall']['score']:.2f}")
                        st.write(f"Gunning Fog: {readability['gunning_fog']['score']:.2f}")
                        st.write(f"Flesch: {readability['flesch']['score']:.2f}")
                    
                    # Add coherence details if available with enhanced visualization
                    if 'details' in selected_result['coherence'] and selected_result['coherence']['details']:
                        st.subheader("Coherence Between Paragraphs")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Create a DataFrame for better display
                            detail_data = []
                            for detail in selected_result['coherence']['details']:
                                # Handle both new and old format
                                if 'from_paragraph' in detail:
                                    detail_data.append({
                                        "From": f"P{detail['from_paragraph']+1}",
                                        "To": f"P{detail['to_paragraph']+1}",
                                        "Score": detail['score'],
                                        "Quality": detail['quality']
                                    })
                                else:
                                    detail_data.append({
                                        "Paragraphs": f"{detail['between'][0]+1} → {detail['between'][1]+1}",
                                        "Score": detail['score'],
                                        "Quality": detail['quality']
                                    })
                            
                            if detail_data:
                                detail_df = pd.DataFrame(detail_data)
                                st.dataframe(detail_df, use_container_width=True)
                        
                        with col2:
                            # Display transition quality summary if available
                            if selected_result['coherence'].get('transition_quality_summary'):
                                # Create pie chart for transition quality distribution
                                pie_data = pd.DataFrame({
                                    'Quality': selected_result['coherence']['transition_quality_summary'].keys(),
                                    'Count': selected_result['coherence']['transition_quality_summary'].values()
                                })
                                
                                # Use custom colors for different quality levels
                                colors = {
                                    'High': '#2ca02c',     # Green
                                    'Medium': '#1f77b4',   # Blue
                                    'Low': '#ff7f0e',      # Orange
                                    'Very Low': '#d62728'  # Red
                                }
                                
                                fig = px.pie(
                                    pie_data, 
                                    values='Count', 
                                    names='Quality',
                                    color='Quality',
                                    color_discrete_map=colors,
                                    title="Transition Quality Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                with tabs[3]:
                    st.json(selected_result)
                
                # Download button for selected result
                result_json = json.dumps(selected_result, indent=2)
                st.download_button(
                    label="Download This Evaluation as JSON",
                    data=result_json,
                    file_name=f"evaluation_result_{selected_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Export all history
        if len(history_data) > 0:
            all_results_json = json.dumps(st.session_state.evaluation_results, indent=2)
            st.download_button(
                label="Download All Evaluation History as JSON",
                data=all_results_json,
                file_name=f"all_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Option to clear history
            if st.button("Clear Evaluation History"):
                st.session_state.evaluation_results = []
                st.experimental_rerun()

elif selected_tab == "Metrics Explanation":
    st.title("Metrics Explanation")
    
    st.markdown("""
    ## Evaluation Metrics
    
    This dashboard uses several metrics to evaluate text quality:

    ### Coherence
    Evaluates the logical flow and connection between paragraphs and ideas in the text.
    
    - **Coherence Score**: Overall score (0-10) representing how well paragraphs connect to each other
    - **Quality Levels**: 
      - **High** (0.7-1.0): Strong connections between paragraphs with clear flow
      - **Medium** (0.5-0.7): Adequate connections but some transitions could be improved
      - **Low** (0.3-0.5): Weak connections with noticeable jumps between ideas
      - **Very Low** (0.0-0.3): Poor connections with abrupt or confusing transitions
    
    - **Transition Quality Distribution**: Shows the proportion of transitions at each quality level
    - **Paragraph Coherence Details**: Provides scores for each paragraph-to-paragraph transition
    
    ### Readability Metrics
    #### Dale-Chall Score
    Calculates readability based on sentence length and percentage of difficult words.
    - Below 4.9: Very easy (Grade 4 and below)
    - 5.0-5.9: Easy (Grades 5-6)
    - 6.0-6.9: Fairly easy (Grades 7-8)
    - 7.0-7.9: Standard (Grades 9-10)
    - 8.0-8.9: Fairly difficult (Grades 11-12)
    - 9.0-9.9: Difficult (College)
    - Above 10.0: Very difficult (College graduate)
    
    #### Gunning Fog Index
    Estimates the years of formal education needed to understand the text.
    - Below 6: Very easy to read
    - 6-8: Easy to read
    - 8-10: Fairly easy to read
    - 10-12: Standard/Plain English
    - 12-15: Fairly difficult to read
    - 15-18: Difficult to read
    - Above 18: Very difficult to read
    

    
    ### Safety/Toxicity
    Evaluates potential harmful content on a scale from 0 to 1:
    - 0.0-0.3: Minor or no safety concerns
    - 0.3-0.7: Moderate safety concerns
    - 0.7-1.0: Significant safety concerns
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© Text Evaluation Dashboard")
