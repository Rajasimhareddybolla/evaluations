import streamlit as st
from full_eval import STEMEvaluator
import json

# Initialize the evaluator
evaluator = STEMEvaluator()

# Streamlit app title
st.title("STEM Explanation Evaluator")

# Sidebar for evaluation type selection
evaluation_type = st.sidebar.selectbox(
    "Select Evaluation Type",
    ["Ground Truth Evaluation", "RAG Evaluation"]
)

# Common inputs
explanation = st.text_area("Explanation to Evaluate", placeholder="Enter the explanation here...")
subject = st.text_input("Subject", placeholder="e.g., Mathematics, Physics, Coding")

if evaluation_type == "Ground Truth Evaluation":
    ground_truth = st.text_area("Ground Truth Explanation", placeholder="Enter the ground truth explanation here...")
    question = st.text_input("Question (Optional)", placeholder="Enter the question here...")
    if st.button("Evaluate"):
        result = evaluator.evaluate_with_ground_truth(
            explanation=explanation,
            ground_truth=ground_truth,
            subject=subject,
            question=question
        )
        st.subheader("Evaluation Result")
        st.json(result)

elif evaluation_type == "RAG Evaluation":
    rag_material = st.text_area("Reference Material", placeholder="Enter the reference material here...")
    question = st.text_input("Question (Optional)", placeholder="Enter the question here...")
    if st.button("Evaluate"):
        result = evaluator.evaluate_with_rag(
            explanation=explanation,
            rag_material=rag_material,
            subject=subject,
            question=question
        )
        st.subheader("Evaluation Result")
        st.json(result)

elif evaluation_type == "Coding Solution Evaluation":
    test_cases = st.text_area("Test Cases (JSON Format)", placeholder='[{"input": "test input", "output": "expected output"}]')
    expected_output = st.text_area("Expected Output", placeholder="Enter the expected output here...")
    actual_output = st.text_area("Actual Output", placeholder="Enter the actual output here...")
    if st.button("Evaluate"):
        try:
            test_cases = json.loads(test_cases)
            result = evaluator.evaluate_coding_solution(
                explanation=explanation,
                test_cases=test_cases,
                expected_output=expected_output,
                actual_output=actual_output,
                subject=subject
            )
            st.subheader("Evaluation Result")
            st.json(result)
        except json.JSONDecodeError:
            st.error("Invalid JSON format for test cases.")

elif evaluation_type == "Web Search Evaluation":
    question = st.text_input("Question (Optional)", placeholder="Enter the question here...")
    if st.button("Evaluate"):
        result = evaluator.evaluate_with_web_search(
            explanation=explanation,
            subject=subject,
            question=question
        )
        st.subheader("Evaluation Result")
        st.json(result)
