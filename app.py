import streamlit as st
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Add the project directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the evaluator
from eval2 import ImprovedSTEMEvaluator

# Set page configuration
st.set_page_config(
    page_title="MIOO STEM Evaluator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .system-prompt {
        background-color: #f9f9f9;
        border-left: 3px solid #1f77b4;
        padding: 10px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 0.9em;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .highlight {
        background-color: #ffffcc;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for system prompts
if 'system_prompts' not in st.session_state:
    st.session_state.system_prompts = {
        'accuracy': """
        You are an expert accuracy evaluator for STEM content specializing in detecting factual errors
        and misconceptions. Your task is to rigorously assess the accuracy of explanations by examining
        factual correctness, conceptual precision, and alignment with established scientific/mathematical
        principles.
        
        Evaluation guidelines:
        1. Focus on identifying factual errors, incorrect formulas/equations, or scientific misconceptions
        2. Check mathematical calculations and logical reasoning for correctness
        3. Evaluate precision of terminology and definitions
        4. Consider domain-specific accuracy standards for the subject
        5. If reference material is provided, compare against it for factual alignment
        
        Score on a scale of 1-10 where:
        - 1-3: Poor (contains critical errors that would significantly mislead a learner)
        - 4-6: Fair (contains minor errors or imprecisions that partially affect understanding)
        - 7-8: Good (generally accurate with very minor imprecisions)
        - 9-10: Excellent (completely accurate, precise, and scientifically/mathematically sound)
        """,
        
        'completeness': """
        You are an expert STEM content evaluator specializing in assessing the completeness of explanations.
        Your task is to determine if an explanation covers all necessary aspects of a topic or question.
        
        Evaluation guidelines:
        1. Identify any missing key concepts, steps, or components essential to the explanation
        2. Check if all parts of a multi-part question/problem are addressed
        3. Evaluate whether contextual information and background knowledge are sufficiently provided
        4. Assess if appropriate examples, applications, or implications are included
        5. Consider depth of coverage relative to the complexity of the subject
        
        Score on a scale of 1-10 where:
        - 1-3: Poor (major components missing, severely incomplete)
        - 4-6: Fair (addresses main points but omits some important elements)
        - 7-8: Good (covers most necessary elements with minor omissions)
        - 9-10: Excellent (comprehensive coverage of all relevant concepts and components)
        """,
        
        'coherence': """
        You are an expert in evaluating the logical flow and coherence of STEM explanations.
        Your task is to assess how well ideas connect, the logical progression of concepts,
        and the overall flow of an explanation.
        
        Evaluation guidelines:
        1. Evaluate logical flow and transitions between ideas
        2. Check for appropriate sequencing of concepts (simple to complex, prerequisites before advanced topics)
        3. Assess consistent terminology and notation throughout
        4. Identify any contradictions, non-sequiturs, or abrupt topic shifts
        5. Consider if the explanation maintains a clear narrative thread
        
        Score on a scale of 1-10 where:
        - 1-3: Poor (disjointed, illogical jumps, contradictions)
        - 4-6: Fair (somewhat logical but with flow issues)
        - 7-8: Good (generally coherent with minor flow issues)
        - 9-10: Excellent (seamless logical flow, excellent transitions)
        """,
        
        'readability': """
        You are an expert in evaluating the readability of STEM explanations specifically for LLM users.
        Your task is to assess how accessible and understandable the text is for the intended audience.
        
        Evaluation guidelines:
        1. Evaluate sentence complexity and length appropriateness
        2. Check for clear definitions of technical terms when first introduced
        3. Assess the balance between technical precision and accessibility
        4. Consider appropriate complexity level for the intended audience
        5. Evaluate the use of concrete examples to illustrate abstract concepts
        
        Score on a scale of 1-10 where:
        - 1-3: Poor (unnecessarily complex, highly technical without sufficient explanation)
        - 4-6: Fair (somewhat accessible but with readability issues)
        - 7-8: Good (generally readable with appropriate terminology)
        - 9-10: Excellent (optimally readable for intended audience while maintaining accuracy)
        """,
        
        'clarity': """
        You are an expert in evaluating the clarity of STEM explanations. Your task is to assess
        how clear, well-structured, and effectively communicated an explanation is.
        
        Evaluation guidelines:
        1. Evaluate the structural organization and logical presentation
        2. Check for clear introduction of concepts and natural progression
        3. Assess use of examples, analogies, or visualizations to illustrate concepts
        4. Consider precision and conciseness of language
        5. Evaluate whether the explanation avoids unnecessary jargon or explains technical terms
        
        Score on a scale of 1-10 where:
        - 1-3: Poor (confusing, disorganized, obscure)
        - 4-6: Fair (somewhat clear but with organizational issues)
        - 7-8: Good (mostly clear with minor issues)
        - 9-10: Excellent (exceptionally clear, well-structured, and effectively communicated)
        """,
        
        'safety': """
        You are an expert in evaluating STEM educational content for safety and ethical considerations.
        Your task is to identify any potentially harmful, misleading, or ethically problematic content.
        
        Evaluation guidelines:
        1. Identify any unsafe experimental procedures or dangerous advice
        2. Flag any misinformation that could lead to harmful outcomes
        3. Check for biased or discriminatory content
        4. Identify pseudoscientific claims presented as factual
        5. Note any content that could promote dangerous activities
        
        Analyze the explanation and identify any safety concerns. Be specific about what aspects
        are problematic and why they raise safety or ethical issues.
        """
    }

# Initialize evaluator with empty instance
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None

# Header
st.title("🔬 MIOO STEM Evaluator")
st.markdown("Comprehensive evaluation of STEM explanations using AI and ML techniques")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_name = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-pro"],
    index=0
)

# Subject selection
subject = st.sidebar.text_input("Subject", "General STEM")

# Target audience
target_audience = st.sidebar.selectbox(
    "Target Audience",
    ["general", "elementary", "middle school", "high school", "undergraduate", "graduate", "expert"],
    index=0
)

# Weights configuration
st.sidebar.subheader("Evaluation Weights")
st.sidebar.markdown("Adjust the importance of each evaluation metric:")

accuracy_weight = st.sidebar.slider("Accuracy", 0.1, 0.5, 0.30, 0.05)
completeness_weight = st.sidebar.slider("Completeness", 0.1, 0.4, 0.20, 0.05)
coherence_weight = st.sidebar.slider("Coherence", 0.05, 0.3, 0.15, 0.05)
clarity_weight = st.sidebar.slider("Clarity", 0.05, 0.3, 0.15, 0.05)
readability_weight = st.sidebar.slider("Readability", 0.05, 0.3, 0.10, 0.05)
safety_weight = st.sidebar.slider("Safety", 0.05, 0.3, 0.10, 0.05)

# Normalize weights to sum to 1
total_weight = accuracy_weight + completeness_weight + coherence_weight + clarity_weight + readability_weight + safety_weight
accuracy_weight = round(accuracy_weight / total_weight, 2)
completeness_weight = round(completeness_weight / total_weight, 2)
coherence_weight = round(coherence_weight / total_weight, 2)
clarity_weight = round(clarity_weight / total_weight, 2)
readability_weight = round(readability_weight / total_weight, 2)
safety_weight = round(safety_weight / total_weight, 2)

# Show normalized weights
st.sidebar.markdown("**Normalized Weights:**")
st.sidebar.markdown(f"- Accuracy: {accuracy_weight:.2f}")
st.sidebar.markdown(f"- Completeness: {completeness_weight:.2f}")
st.sidebar.markdown(f"- Coherence: {coherence_weight:.2f}")
st.sidebar.markdown(f"- Clarity: {clarity_weight:.2f}")
st.sidebar.markdown(f"- Readability: {readability_weight:.2f}")
st.sidebar.markdown(f"- Safety: {safety_weight:.2f}")

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    st.markdown("Lower temperature = more consistent results")
    
    # System prompts customization
    st.subheader("System Prompts")
    st.markdown("Customize the instructions for each evaluation dimension:")
    
    # Create tabs for each system prompt
    prompt_tabs = st.tabs(["Accuracy", "Completeness", "Coherence", "Readability", "Clarity", "Safety"])
    
    with prompt_tabs[0]:
        st.session_state.system_prompts['accuracy'] = st.text_area(
            "Accuracy System Prompt", 
            st.session_state.system_prompts['accuracy'],
            height=300
        )
    
    with prompt_tabs[1]:
        st.session_state.system_prompts['completeness'] = st.text_area(
            "Completeness System Prompt", 
            st.session_state.system_prompts['completeness'],
            height=300
        )
    
    with prompt_tabs[2]:
        st.session_state.system_prompts['coherence'] = st.text_area(
            "Coherence System Prompt", 
            st.session_state.system_prompts['coherence'],
            height=300
        )
    
    with prompt_tabs[3]:
        st.session_state.system_prompts['readability'] = st.text_area(
            "Readability System Prompt", 
            st.session_state.system_prompts['readability'],
            height=300
        )
    
    with prompt_tabs[4]:
        st.session_state.system_prompts['clarity'] = st.text_area(
            "Clarity System Prompt", 
            st.session_state.system_prompts['clarity'],
            height=300
        )
    
    with prompt_tabs[5]:
        st.session_state.system_prompts['safety'] = st.text_area(
            "Safety System Prompt", 
            st.session_state.system_prompts['safety'],
            height=300
        )

# Main content area
tab1, tab2 = st.tabs(["Evaluation", "How It Works"])

with tab1:
    # Input section
    st.header("Input")
    col1, col2 = st.columns(2)
    
    with col1:
        explanation = st.text_area(
            "Enter STEM explanation to evaluate",
            height=250,
            placeholder="Enter the STEM explanation text here..."
        )
        
        question = st.text_input("Question (optional)", 
                                placeholder="What specific question does this explanation address?")
    
    with col2:
        reference = st.text_area(
            "Reference material (optional)",
            height=250,
            placeholder="Enter reference material or ground truth for comparison..."
        )
        
        st.info("Reference material helps evaluate accuracy and completeness, but is not required.")
    
    # Initialize/update evaluator with current settings
    if st.button("Evaluate", type="primary"):
        if not explanation:
            st.error("Please enter an explanation to evaluate.")
        else:
            with st.spinner("Evaluating... This may take a moment."):
                # Create a new evaluator with current settings
                st.session_state.evaluator = ImprovedSTEMEvaluator(
                    model_name=model_name,
                    temperature=temperature
                )
                
                # Save the original system prompts
                original_prompts = {}
                for metric, prompt in st.session_state.system_prompts.items():
                    original_prompts[metric] = getattr(st.session_state.evaluator, f"_create_gemini_response_schema")(metric)
                
                # Add custom weights
                custom_weights = {
                    "accuracy": accuracy_weight,
                    "completeness": completeness_weight,
                    "coherence": coherence_weight,
                    "clarity": clarity_weight,
                    "readability": readability_weight,
                    "safety": safety_weight
                }
                
                # Run evaluation
                try:
                    results = st.session_state.evaluator.comprehensive_evaluation(
                        explanation=explanation,
                        reference=reference if reference else None,
                        question=question if question else None,
                        subject=subject,
                        target_audience=target_audience
                    )
                    
                    # Recalculate the overall score with custom weights
                    scores = {}
                    for metric, weight in custom_weights.items():
                        if metric in results['scores']:
                            scores[metric] = results['scores'][metric]
                    
                    weighted_score = sum(scores[metric] * weight for metric, weight in custom_weights.items())
                    results['overall_score'] = round(weighted_score, 2)
                    results['overall_percentage'] = round(weighted_score * 10, 2)
                    results['weights'] = custom_weights
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.explanation = explanation
                    st.session_state.reference = reference
                    st.session_state.question = question
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    st.exception(e)
    
    # Results display
    if 'results' in st.session_state:
        st.header("Evaluation Results")
        
        results = st.session_state.results
        
        # Display overall score with gauge
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Create overall score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results['overall_percentage'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Score", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "royalblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'lightcoral'},
                        {'range': [50, 75], 'color': 'lightyellow'},
                        {'range': [75, 100], 'color': 'lightgreen'},
                    ],
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="white",
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"### Summary")
            st.markdown(results['summary'])
        
        with col2:
            # Display individual scores as horizontal bars
            categories = list(results['scores'].keys())
            scores = list(results['scores'].values())
            weights = [results['weights'][cat] for cat in categories]
            
            # Format category names
            categories = [cat.capitalize() for cat in categories]
            
            # Create horizontal bar chart with custom colors based on score
            colors = ['#ff9999' if s < 5 else '#ffcc99' if s < 7 else '#99cc99' for s in scores]
            
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                y=categories,
                x=scores,
                orientation='h',
                marker_color=colors,
                text=[f"{s}/10 (weight: {w:.2f})" for s, w in zip(scores, weights)],
                textposition='auto',
                name='Score'
            ))
            
            # Update layout
            fig.update_layout(
                title="Dimension Scores",
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(
                    title="Score (0-10)",
                    range=[0, 10],
                    dtick=1
                ),
                yaxis=dict(
                    title="",
                    autorange="reversed"
                ),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results tabs
        metric_tabs = st.tabs([
            "Accuracy", "Completeness", "Coherence", 
            "Readability", "Clarity", "Safety"
        ])
        
        # Accuracy tab
        with metric_tabs[0]:
            if 'accuracy' in results['detailed_evaluations']:
                accuracy_result = results['detailed_evaluations']['accuracy']
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"### Accuracy Score: {accuracy_result.get('accuracy_score', 'N/A')}/10")
                    st.markdown("#### Analysis")
                    st.markdown(accuracy_result.get('analysis', 'No analysis available'))
                
                with col2:
                    st.markdown("#### Strengths")
                    for strength in accuracy_result.get('strengths', []):
                        st.markdown(f"✅ {strength}")
                    
                    st.markdown("#### Weaknesses")
                    for weakness in accuracy_result.get('weaknesses', []):
                        st.markdown(f"❌ {weakness}")
                
                st.markdown("#### Suggestions for Improvement")
                for suggestion in accuracy_result.get('suggestions', []):
                    st.markdown(f"💡 {suggestion}")
                
                with st.expander("System Prompt Used"):
                    st.markdown(f"```\n{st.session_state.system_prompts['accuracy']}\n```")
        
        # Completeness tab
        with metric_tabs[1]:
            if 'completeness' in results['detailed_evaluations']:
                completeness_result = results['detailed_evaluations']['completeness']
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"### Completeness Score: {completeness_result.get('completeness_score', 'N/A')}/10")
                    st.markdown("#### Analysis")
                    st.markdown(completeness_result.get('analysis', 'No analysis available'))
                
                with col2:
                    st.markdown("#### Strengths")
                    for strength in completeness_result.get('strengths', []):
                        st.markdown(f"✅ {strength}")
                    
                    st.markdown("#### Weaknesses")
                    for weakness in completeness_result.get('weaknesses', []):
                        st.markdown(f"❌ {weakness}")
                
                st.markdown("#### Suggestions for Improvement")
                for suggestion in completeness_result.get('suggestions', []):
                    st.markdown(f"💡 {suggestion}")
                
                with st.expander("System Prompt Used"):
                    st.markdown(f"```\n{st.session_state.system_prompts['completeness']}\n```")
        
        # Coherence tab
        with metric_tabs[2]:
            if 'coherence' in results['detailed_evaluations']:
                coherence_result = results['detailed_evaluations']['coherence']
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"### Coherence Score: {coherence_result.get('coherence_score', 'N/A')}/10")
                    st.markdown(f"#### Final Score (weighted): {coherence_result.get('final_coherence_score', 'N/A')}/10")
                    st.markdown("#### Analysis")
                    st.markdown(coherence_result.get('analysis', 'No analysis available'))
                
                with col2:
                    st.markdown("#### Strengths")
                    for strength in coherence_result.get('strengths', []):
                        st.markdown(f"✅ {strength}")
                    
                    st.markdown("#### Weaknesses")
                    for weakness in coherence_result.get('weaknesses', []):
                        st.markdown(f"❌ {weakness}")
                
                st.markdown("#### Suggestions for Improvement")
                for suggestion in coherence_result.get('suggestions', []):
                    st.markdown(f"💡 {suggestion}")
                
                # ML Coherence analysis
                st.markdown("### ML Coherence Analysis")
                st.markdown(f"ML-calculated coherence score: {coherence_result.get('ml_coherence_score', 'N/A')}")
                
                # Paragraph transitions visualization if available
                if 'paragraph_transitions' in coherence_result and coherence_result['paragraph_transitions']:
                    transitions = coherence_result['paragraph_transitions']
                    
                    # Create a dataframe for the transitions
                    transition_data = []
                    for t in transitions:
                        transition_data.append({
                            'From': f"Para {t['paragraphs'][0]}",
                            'To': f"Para {t['paragraphs'][1]}",
                            'Score': t['score']
                        })
                    
                    df = pd.DataFrame(transition_data)
                    
                    # Display as table
                    st.markdown("#### Paragraph Transitions")
                    st.dataframe(df)
                    
                    # Visualize as heatmap
                    if len(transitions) > 1:
                        # Create a matrix for the heatmap
                        num_paragraphs = max([t['paragraphs'][1] for t in transitions])
                        matrix = np.zeros((num_paragraphs, num_paragraphs))
                        
                        for t in transitions:
                            i, j = t['paragraphs'][0]-1, t['paragraphs'][1]-1
                            matrix[i, j] = t['score']
                        
                        # Plot heatmap
                        fig, ax = plt.subplots(figsize=(5, 5))
                        im = ax.imshow(matrix, cmap='YlGnBu')
                        
                        # Add labels
                        ax.set_xticks(np.arange(num_paragraphs))
                        ax.set_yticks(np.arange(num_paragraphs))
                        ax.set_xticklabels([f"Para {i+1}" for i in range(num_paragraphs)])
                        ax.set_yticklabels([f"Para {i+1}" for i in range(num_paragraphs)])
                        
                        # Rotate x labels
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                        
                        # Add colorbar
                        cbar = ax.figure.colorbar(im, ax=ax)
                        cbar.ax.set_ylabel("Coherence Score", rotation=-90, va="bottom")
                        
                        # Loop over data to add text annotations
                        for i in range(num_paragraphs):
                            for j in range(num_paragraphs):
                                if matrix[i, j] > 0:
                                    text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                                ha="center", va="center", color="black")
                        
                        ax.set_title("Paragraph Coherence Matrix")
                        fig.tight_layout()
                        
                        st.pyplot(fig)
                
                with st.expander("System Prompt Used"):
                    st.markdown(f"```\n{st.session_state.system_prompts['coherence']}\n```")
        
        # Readability tab
        with metric_tabs[3]:
            if 'readability' in results['detailed_evaluations']:
                readability_result = results['detailed_evaluations']['readability']
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"### Readability Score: {readability_result.get('readability_score', 'N/A')}/10")
                    st.markdown("#### Analysis")
                    st.markdown(readability_result.get('analysis', 'No analysis available'))
                
                with col2:
                    st.markdown("#### Strengths")
                    for strength in readability_result.get('strengths', []):
                        st.markdown(f"✅ {strength}")
                    
                    st.markdown("#### Weaknesses")
                    for weakness in readability_result.get('weaknesses', []):
                        st.markdown(f"❌ {weakness}")
                
                st.markdown("#### Suggestions for Improvement")
                for suggestion in readability_result.get('suggestions', []):
                    st.markdown(f"💡 {suggestion}")
                
                # Display readability metrics
                if 'metrics' in readability_result:
                    metrics = readability_result['metrics']
                    
                    st.markdown("### Readability Metrics")
                    
                    # Create 3 columns for metrics
                    m_col1, m_col2, m_col3 = st.columns(3)
                    
                    with m_col1:
                        st.metric("Flesch-Kincaid Grade", metrics.get('flesch_kincaid_grade', 'N/A'),
                                 help="Lower values indicate easier reading (U.S. grade level)")
                        st.metric("SMOG Index", metrics.get('smog_index', 'N/A'),
                                 help="Estimates years of education needed to understand text")
                        
                    with m_col2:
                        st.metric("Flesch Reading Ease", metrics.get('flesch_reading_ease', 'N/A'),
                                 help="Higher values indicate easier reading (0-100 scale)")
                        st.metric("Avg Word Length", metrics.get('avg_word_length', 'N/A'),
                                 help="Average number of characters per word")
                    
                    with m_col3:
                        st.metric("Avg Sentence Length", metrics.get('avg_sentence_length', 'N/A'),
                                 help="Average number of words per sentence")
                
                with st.expander("System Prompt Used"):
                    st.markdown(f"```\n{st.session_state.system_prompts['readability']}\n```")
        
        # Clarity tab
        with metric_tabs[4]:
            if 'clarity' in results['detailed_evaluations']:
                clarity_result = results['detailed_evaluations']['clarity']
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"### Clarity Score: {clarity_result.get('clarity_score', 'N/A')}/10")
                    st.markdown("#### Analysis")
                    st.markdown(clarity_result.get('analysis', 'No analysis available'))
                
                with col2:
                    st.markdown("#### Strengths")
                    for strength in clarity_result.get('strengths', []):
                        st.markdown(f"✅ {strength}")
                    
                    st.markdown("#### Weaknesses")
                    for weakness in clarity_result.get('weaknesses', []):
                        st.markdown(f"❌ {weakness}")
                
                st.markdown("#### Suggestions for Improvement")
                for suggestion in clarity_result.get('suggestions', []):
                    st.markdown(f"💡 {suggestion}")
                
                with st.expander("System Prompt Used"):
                    st.markdown(f"```\n{st.session_state.system_prompts['clarity']}\n```")
        
        # Safety tab
        with metric_tabs[5]:
            if 'safety' in results['detailed_evaluations']:
                safety_result = results['detailed_evaluations']['safety']
                
                st.markdown(f"### Safety Score: {safety_result.get('safety_score', 'N/A')}/10")
                
                # Safety metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Normalized Score", safety_result.get('normalized_score', 'N/A'))
                    
                    if 'is_safe' in safety_result:
                        if safety_result['is_safe']:
                            st.success("Content appears safe and appropriate")
                        else:
                            st.warning("Content may have safety concerns")
                
                with col2:
                    if 'toxicity_score' in safety_result and safety_result['toxicity_score'] != 'N/A':
                        st.metric("Toxicity Score", safety_result.get('toxicity_score', 'N/A'), 
                                 help="Lower is better (0-1 scale)")
                
                # Safety concerns
                if 'safety_concerns' in safety_result and safety_result['safety_concerns']:
                    st.markdown("### Safety Concerns")
                    for concern in safety_result['safety_concerns']:
                        st.error(concern)
                
                # Flags
                if 'flags' in safety_result and safety_result['flags']:
                    st.markdown("### Flags")
                    for flag in safety_result['flags']:
                        st.warning(flag)
                
                st.markdown("### Analysis")
                st.markdown(safety_result.get('analysis', 'No analysis available'))
                
                with st.expander("System Prompt Used"):
                    st.markdown(f"```\n{st.session_state.system_prompts['safety']}\n```")

with tab2:
    st.header("How the Evaluation Works")
    
    st.markdown("""
    ### Overview
    
    The MIOO STEM Evaluator performs a comprehensive evaluation of STEM explanations across six key dimensions:
    
    1. **Accuracy** - Factual correctness and scientific precision
    2. **Completeness** - Coverage of all necessary concepts and components
    3. **Coherence** - Logical flow and connectivity between ideas
    4. **Readability** - Accessibility and clarity of language for the target audience
    5. **Clarity** - Structure, organization, and effective communication
    6. **Safety/Toxicity** - Absence of harmful or inappropriate content
    
    ### Evaluation Process
    
    Each dimension is evaluated through a combination of:
    
    1. **LLM-based Assessment**: Gemini models with specialized system prompts analyze the text
    2. **ML Techniques**: Sentence embeddings measure coherence between paragraphs
    3. **Traditional NLP Metrics**: Readability formulas like Flesch-Kincaid assess text complexity
    4. **Safety Models**: Specialized toxicity detection models identify potentially harmful content
    
    ### Behind the Scenes
    
    The evaluation follows these steps:
    
    1. **Input Processing**: Text is analyzed and prepared for evaluation
    2. **Parallel Evaluation**: All dimensions are evaluated simultaneously for efficiency
    3. **Metric Calculation**: Each dimension receives a score from 1-10
    4. **Weighted Combination**: Individual scores are weighted and combined for the overall score
    5. **Detailed Analysis**: Strengths, weaknesses, and suggestions are identified for each dimension
    
    ### System Prompts
    
    Each dimension uses a specialized system prompt that instructs the Gemini model how to evaluate that specific aspect. These prompts include:
    
    - Evaluation guidelines and criteria
    - Scoring scale definitions
    - Instructions for identifying strengths and weaknesses
    - Required focus areas for that dimension
    
    You can customize these prompts in the Advanced Settings section.
    
    ### ML Enhancements
    
    The evaluator enhances LLM-based assessments with:
    
    - **Sentence Embeddings**: Vector representations measure semantic similarity between paragraphs
    - **Readability Formulas**: Multiple traditional metrics assess text complexity
    - **Statistical Analysis**: Text statistics provide additional insights
    
    ### Customization
    
    You can customize the evaluation by:
    
    - Adjusting the weights of different dimensions
    - Modifying system prompts for specialized evaluations
    - Changing the target audience for readability assessment
    - Providing reference material for accuracy and completeness evaluation
    """)

    st.subheader("Architecture Diagram")
    
    # Create architecture diagram
    st.markdown("""
    ```
    ┌─────────────────────────────────────┐
    │            Input Text               │
    └───────────────┬─────────────────────┘
                    ▼
    ┌─────────────────────────────────────┐
    │         Parallel Evaluation         │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
    │  │Acc. │ │Comp.│ │Coh. │ │Read.│   │
    │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
    │  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐   │
    │  │Clar.│ │Safe.│ │ ML  │ │Stats│   │
    │  └─────┘ └─────┘ └─────┘ └─────┘   │
    └───────────────┬─────────────────────┘
                    ▼
    ┌─────────────────────────────────────┐
    │        Score Aggregation            │
    │     (Weighted Combination)          │
    └───────────────┬─────────────────────┘
                    ▼
    ┌─────────────────────────────────────┐
    │       Detailed Analysis             │
    │  - Strengths & Weaknesses           │
    │  - Improvement Suggestions          │
    │  - Visualization                    │
    └─────────────────────────────────────┘
    ```
    """)
    
    # Source code section
    with st.expander("View Core Evaluation Code"):
        st.code("""
def comprehensive_evaluation(self, 
                          explanation: str, 
                          reference: Optional[str] = None,
                          question: Optional[str] = None,
                          subject: str = "General STEM",
                          target_audience: str = "general") -> Dict:
    \"\"\"
    Perform a comprehensive evaluation using all six metrics.
    \"\"\"
    # Create a thread-safe dictionary to store results
    evaluation_results = {}
    result_lock = threading.Lock()
    
    def run_evaluation(name, func, *args, **kwargs):
        result = func(*args, **kwargs)
        with result_lock:
            evaluation_results[name] = result
    
    # Define evaluation tasks
    tasks = [
        ("accuracy", self.evaluate_accuracy, explanation, reference, question, subject),
        ("completeness", self.evaluate_completeness, explanation, reference, question, subject),
        ("coherence", self.evaluate_coherence, explanation),
        ("readability", self.evaluate_readability, explanation, target_audience),
        ("clarity", self.evaluate_clarity, explanation, question, subject),
        ("safety", self.evaluate_safety, explanation)
    ]
    
    # Run evaluations in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(run_evaluation, name, func, *args) 
                  for name, func, *args in tasks]
        # Wait for all futures to complete
        for future in futures:
            future.result()
    
    # Calculate overall score with weighted metrics
    weights = {
        "accuracy": 0.30,      # Most important for STEM content
        "completeness": 0.20,  # Second most important
        "coherence": 0.15,     # Important for logical flow
        "clarity": 0.15,       # Equally important for understanding
        "readability": 0.10,   # Important but less critical
        "safety": 0.10         # Important for responsible content
    }
    
    scores = {}
    for metric, weight in weights.items():
        if metric in evaluation_results:
            # Get the score, defaulting to 0 if not found
            score_key = f"{metric}_score"
            if metric == "coherence" and "final_coherence_score" in evaluation_results[metric]:
                scores[metric] = evaluation_results[metric]["final_coherence_score"]
            elif score_key in evaluation_results[metric]:
                scores[metric] = evaluation_results[metric][score_key]
            else:
                scores[metric] = 0
        else:
            scores[metric] = 0
    
    # Calculate weighted score
    weighted_score = sum(scores[metric] * weight for metric, weight in weights.items())
    
    # Prepare comprehensive result
    comprehensive_result = {
        "overall_score": round(weighted_score, 2),
        "overall_percentage": round(weighted_score * 10, 2),  # Convert to percentage
        "scores": {metric: round(score, 2) for metric, score in scores.items()},
        "weights": weights,
        "detailed_evaluations": evaluation_results,
        "summary": self._generate_evaluation_summary(
            explanation, scores, weighted_score, question, subject)
    }
    
    return comprehensive_result
        """, language="python")

# Run the app
if __name__ == "__main__":
    pass
