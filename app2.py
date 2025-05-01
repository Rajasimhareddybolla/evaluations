import streamlit as st
import pandas as pd
import json
import concurrent.futures
from utils import utils as EvalUtils, Gemini, Accuracy, c_c_prompt, saftey # Renamed utils to EvalUtils to avoid conflict
import time
import os
import logging # Import logging

# --- Configuration and Initialization ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Set page config
st.set_page_config(layout="wide", page_title="LLM Evaluation Dashboard")

# Load API Key (Ensure GEMINI_API_KEY is set as an environment variable or use st.secrets)
# Make sure to set the GEMINI_API_KEY environment variable before running the script
# Example: export GEMINI_API_KEY='your_api_key_here'
if "GEMINI_API_KEY" not in os.environ:
    st.error("GEMINI_API_KEY environment variable not set. Please set it to run the LLM evaluations.")
    st.stop()

@st.cache_resource # Cache the utils instance as it loads the sentence model
def get_eval_utils():
    """Initializes and returns the evaluation utilities class."""
    return EvalUtils()

@st.cache_resource # Cache the Gemini client
def get_gemini_client():
    """Initializes and returns the Gemini client."""
    try:
        return Gemini()
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        return None

eval_utils = get_eval_utils()
gemini_client = get_gemini_client()

# --- Caching for LLM Calls ---
# Simple dictionary-based cache for demonstration.
# For production, consider more robust caching like diskcache or joblib.
llm_cache = {}

def generate_with_cache(cache_key, prompt, temperature=0, model_name="gemini-1.5-flash"):
    """Generates content using Gemini API with simple caching. Returns (response_text, status_message, error_message)."""
    if cache_key in llm_cache:
        logger.info(f"Cache hit for: {cache_key}")
        return llm_cache[cache_key], "Cache hit", None

    if gemini_client:
        try:
            logger.info(f"Cache miss. Calling Gemini API for: {cache_key}")
            response_text = gemini_client.generate(prompt, temperature, model_name)
            llm_cache[cache_key] = response_text # Store raw text
            return response_text, "API Call Success", None
        except Exception as e:
            logger.error(f"Error calling Gemini API for {cache_key}: {e}")
            return None, "API Call Error", f"Error calling Gemini API for {cache_key}: {e}"
    else:
        logger.warning("Gemini client not available. Skipping LLM call.")
        return None, "Skipped", "Gemini client not available. Skipping LLM call."

# --- Evaluation Functions ---

def evaluate_accuracy(input_text, output_text, context_text):
    """Formats prompt and calls LLM for accuracy evaluation. Returns (result_dict, prompt, raw_response, status_message, error_message)."""
    prompt_template, _ = Accuracy
    prompt = prompt_template.replace("{{input}}", input_text or "N/A") \
                            .replace("{{output}}", output_text or "N/A") \
                            .replace("{{context}}", context_text or "N/A")
    cache_key = f"accuracy_{hash(prompt)}"
    raw_response, status, error = generate_with_cache(cache_key, prompt)

    if error:
        return {"accuracy": f"Error: {error}"}, prompt, raw_response, status, error

    if raw_response:
        try:
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                 parsed_json = json.loads(raw_response[json_start:json_end])
                 return parsed_json, prompt, raw_response, status, None
            else:
                 warning_msg = f"Accuracy: Could not find valid JSON in response: {raw_response}"
                 logger.warning(warning_msg)
                 return {"accuracy": "Error: Invalid JSON Response"}, prompt, raw_response, status, warning_msg

        except json.JSONDecodeError as e:
            error_msg = f"Accuracy: Failed to parse JSON response: {e}\nRaw response: {raw_response}"
            logger.error(error_msg)
            return {"accuracy": f"Error: {e}"}, prompt, raw_response, status, error_msg
    # This case should ideally be covered by generate_with_cache returning an error
    return {"accuracy": "Error: LLM call failed unexpectedly"}, prompt, raw_response, status, "LLM call failed unexpectedly"


def evaluate_completeness_clarity(input_text, output_text):
    """Formats prompt and calls LLM for completeness and clarity evaluation. Returns (result_dict, prompt, raw_response, status_message, error_message)."""
    prompt_template, _ = c_c_prompt
    prompt = prompt_template.replace("{{input}}", input_text or "N/A") \
                            .replace("{{output}}", output_text or "N/A")
    cache_key = f"cc_{hash(prompt)}"
    raw_response, status, error = generate_with_cache(cache_key, prompt)

    default_error_result = {"completeness": "Error", "clarity": "Error", "relevance": "Error"}

    if error:
        return {**default_error_result, "error_details": error}, prompt, raw_response, status, error

    if raw_response:
        try:
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                 parsed_json = json.loads(raw_response[json_start:json_end])
                 warnings = []
                 for key in ["completeness", "clarity", "relevance"]:
                     if key in parsed_json:
                         try:
                             score_str = str(parsed_json[key]).split('/')[0].split('-')[0].strip()
                             parsed_json[key+'_score'] = float(score_str)
                         except (ValueError, TypeError):
                             parsed_json[key+'_score'] = None
                             warning_msg = f"Completeness/Clarity: Could not convert '{parsed_json[key]}' to float for '{key}'."
                             logger.warning(warning_msg)
                             warnings.append(warning_msg)
                 return parsed_json, prompt, raw_response, status, "; ".join(warnings) if warnings else None
            else:
                 warning_msg = f"Completeness/Clarity: Could not find valid JSON in response: {raw_response}"
                 logger.warning(warning_msg)
                 return {**default_error_result, "error_details": "Invalid JSON Response"}, prompt, raw_response, status, warning_msg

        except json.JSONDecodeError as e:
            error_msg = f"Completeness/Clarity: Failed to parse JSON response: {e}\nRaw response: {raw_response}"
            logger.error(error_msg)
            return {**default_error_result, "error_details": f"JSONDecodeError: {e}"}, prompt, raw_response, status, error_msg

    return {**default_error_result, "error_details": "LLM call failed unexpectedly"}, prompt, raw_response, status, "LLM call failed unexpectedly"


def evaluate_safety(output_text):
    """Formats prompt and calls LLM for safety evaluation. Returns (result_dict, prompt, raw_response, status_message, error_message)."""
    prompt_template, _ = saftey
    prompt = prompt_template.replace("{{output}}", output_text or "N/A")
    cache_key = f"safety_{hash(prompt)}"
    raw_response, status, error = generate_with_cache(cache_key, prompt)

    if error:
        return {"safety_score": f"Error: {error}"}, prompt, raw_response, status, error

    if raw_response:
        try:
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                 parsed_json = json.loads(raw_response[json_start:json_end])
                 score_key = next((k for k in parsed_json if k.strip() == "safety_score"), None)
                 warning_msg = None
                 if score_key:
                     parsed_json["safety_score"] = parsed_json.pop(score_key)
                 else:
                     warning_msg = f"Safety: 'safety_score' key not found in response: {parsed_json}"
                     logger.warning(warning_msg)
                     parsed_json["safety_score"] = None

                 return parsed_json, prompt, raw_response, status, warning_msg
            else:
                 warning_msg = f"Safety: Could not find valid JSON in response: {raw_response}"
                 logger.warning(warning_msg)
                 return {"safety_score": "Error: Invalid JSON Response"}, prompt, raw_response, status, warning_msg

        except json.JSONDecodeError as e:
            error_msg = f"Safety: Failed to parse JSON response: {e}\nRaw response: {raw_response}"
            logger.error(error_msg)
            return {"safety_score": f"Error: {e}"}, prompt, raw_response, status, error_msg

    return {"safety_score": "Error: LLM call failed unexpectedly"}, prompt, raw_response, status, "LLM call failed unexpectedly"


def evaluate_coherence(output_text):
    """Evaluates coherence using the local utils method. Returns (result_dict, error_message)."""
    if not output_text:
        return {"coherence_score": 0, "feedback": "No text provided for coherence evaluation.", "average_coherence": 0, "overall_quality": "N/A"}, None
    try:
        return eval_utils.evaluate_coherence(output_text), None
    except Exception as e:
        error_msg = f"Error during coherence evaluation: {e}"
        logger.error(error_msg)
        return {"coherence_score": "Error", "feedback": str(e)}, error_msg

def evaluate_readability(output_text):
    """Evaluates readability using local utils methods. Returns (result_dict, error_message)."""
    if not output_text:
        return {"dale_chall": "N/A", "gunning_fog": "N/A", "flesch_reading_ease": "N/A"}, None
    results = {}
    errors = []
    try:
        results["dale_chall"] = round(eval_utils.dale_chall(output_text), 2)
    except Exception as e:
        msg = f"Could not calculate Dale-Chall score: {e}"
        logger.warning(msg)
        errors.append(msg)
        results["dale_chall"] = "Error"
    try:
        results["gunning_fog"] = round(eval_utils.gunning_fog_formula(output_text), 2)
    except Exception as e:
        msg = f"Could not calculate Gunning Fog score: {e}"
        logger.warning(msg)
        errors.append(msg)
        results["gunning_fog"] = "Error"
    try:
        results["flesch_reading_ease"] = round(eval_utils.flesch_reading_ease(output_text), 2)
    except Exception as e:
        msg = f"Could not calculate Flesch Reading Ease score: {e}"
        logger.warning(msg)
        errors.append(msg)
        results["flesch_reading_ease"] = "Error"
    return results, "; ".join(errors) if errors else None

# --- Streamlit App UI ---

st.title("📝 LLM Evaluation Dashboard")
st.markdown("Evaluate language model outputs based on various metrics.")

# Initialize session state for storing results
if 'eval_results' not in st.session_state:
    st.session_state.eval_results = None
if 'prompts' not in st.session_state:
    st.session_state.prompts = {}
if 'raw_responses' not in st.session_state:
    st.session_state.raw_responses = {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inputs")
    input_text = st.text_area("Input / Prompt", height=150, key="input_text")
    context_text = st.text_area("Context (Optional)", height=150, key="context_text")

with col2:
    st.subheader("Output")
    output_text = st.text_area("Output / Response from LLM", height=315, key="output_text")

if st.button("🚀 Evaluate", type="primary"):
    if not output_text:
        st.warning("Please provide the 'Output / Response from LLM' to evaluate.")
    # Removed the gemini_client check here as it's handled inside generate_with_cache
    # elif not gemini_client and (not input_text or not context_text):
    #      st.warning("Gemini client not available. LLM-based evaluations (Accuracy, Completeness, Safety) require the client and potentially Input/Context.")
    else:
        start_time = time.time()
        results = {}
        prompts = {}
        raw_responses = {}
        statuses = {} # To store status messages like cache hit/miss
        errors_warnings = {} # To store errors/warnings from functions

        with st.spinner("Running evaluations... This may take a moment."):
            # Use ThreadPoolExecutor for parallel execution
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {}
                # Submit LLM-based tasks
                futures[executor.submit(evaluate_accuracy, input_text, output_text, context_text)] = "accuracy"
                futures[executor.submit(evaluate_completeness_clarity, input_text, output_text)] = "completeness_clarity"
                futures[executor.submit(evaluate_safety, output_text)] = "safety"

                # Submit local tasks
                futures[executor.submit(evaluate_coherence, output_text)] = "coherence"
                futures[executor.submit(evaluate_readability, output_text)] = "readability"

                for future in concurrent.futures.as_completed(futures):
                    metric_name = futures[future]
                    try:
                        result_tuple = future.result()

                        # Unpack results based on function signature
                        if metric_name in ["accuracy", "completeness_clarity", "safety"]:
                            data, prompt, raw_response, status, error_msg = result_tuple
                            results[metric_name] = data
                            prompts[metric_name] = prompt
                            raw_responses[metric_name] = raw_response
                            statuses[metric_name] = status
                            if error_msg: errors_warnings[metric_name] = error_msg
                        elif metric_name in ["coherence", "readability"]:
                             data, error_msg = result_tuple
                             results[metric_name] = data
                             if error_msg: errors_warnings[metric_name] = error_msg
                        else: # Should not happen
                             logger.error(f"Unknown metric name encountered: {metric_name}")


                    except Exception as exc:
                        logger.error(f'{metric_name} generated an exception during future processing: {exc}')
                        results[metric_name] = {"error": str(exc)}
                        errors_warnings[metric_name] = f"Future processing error: {exc}"


        st.session_state.eval_results = results
        st.session_state.prompts = prompts
        st.session_state.raw_responses = raw_responses
        end_time = time.time()
        st.success(f"Evaluations completed in {end_time - start_time:.2f} seconds.")

        # Display errors/warnings collected from the functions
        if errors_warnings:
            st.subheader("⚠️ Issues Encountered During Evaluation")
            for metric, msg in errors_warnings.items():
                st.warning(f"**{metric.replace('_', ' ').title()}:** {msg}")
        # Display statuses (optional, can be verbose)
        # st.subheader("Evaluation Status")
        # for metric, status in statuses.items():
        #     st.info(f"{metric.replace('_', ' ').title()}: {status}")


# --- Display Results ---
if st.session_state.eval_results:
    st.divider()
    st.header("📊 Evaluation Results")

    results = st.session_state.eval_results
    prompts = st.session_state.prompts
    raw_responses = st.session_state.raw_responses

    # Prepare data for DataFrame and Charts
    summary_data = {}
    numeric_scores = {}

    # --- Individual Metric Sections ---
    col1, col2 = st.columns(2)

    with col1:
        # Accuracy
        st.subheader("🎯 Accuracy")
        if "accuracy" in results:
            acc_res = results["accuracy"]
            # Display the dictionary content, which might include error messages now
            st.write(acc_res)
            # Attempt to extract score only if no error string is present
            accuracy_val = acc_res.get("accuracy", "N/A")
            if isinstance(accuracy_val, str) and "Error:" in accuracy_val:
                 summary_data["Accuracy Score"] = accuracy_val # Keep error message
            else:
                try:
                    # Check for specific score key first, fallback to 'accuracy'
                    score_key = next((k for k in acc_res if k.endswith('_score')), 'accuracy')
                    score = float(acc_res.get(score_key, "NaN"))
                    summary_data["Accuracy Score"] = score
                    numeric_scores["Accuracy"] = score * 10 # Scale 0-1 to 0-10
                except (ValueError, TypeError, KeyError):
                    summary_data["Accuracy Score"] = accuracy_val # Display original value if conversion fails


        # Completeness, Clarity, Relevance
        st.subheader("✅ Completeness, Clarity & Relevance")
        if "completeness_clarity" in results:
            cc_res = results["completeness_clarity"]
            # Display potentially modified results (might contain 'Error')
            st.write(f"**Completeness:** {cc_res.get('completeness', 'N/A')}")
            st.write(f"**Clarity:** {cc_res.get('clarity', 'N/A')}")
            st.write(f"**Relevance:** {cc_res.get('relevance', 'N/A')}")
            if "error_details" in cc_res:
                 st.caption(f"Note: {cc_res['error_details']}")

            # Use pre-calculated numeric scores if available and not None
            comp_score = cc_res.get('completeness_score')
            clar_score = cc_res.get('clarity_score')
            rel_score = cc_res.get('relevance_score')
            summary_data["Completeness Score"] = comp_score if comp_score is not None else cc_res.get('completeness', 'N/A')
            summary_data["Clarity Score"] = clar_score if clar_score is not None else cc_res.get('clarity', 'N/A')
            summary_data["Relevance Score"] = rel_score if rel_score is not None else cc_res.get('relevance', 'N/A')
            if comp_score is not None: numeric_scores["Completeness"] = comp_score
            if clar_score is not None: numeric_scores["Clarity"] = clar_score
            if rel_score is not None: numeric_scores["Relevance"] = rel_score


        # Safety
        st.subheader("🛡️ Safety")
        if "safety" in results:
            safety_res = results["safety"]
            st.write(safety_res) # Display dict, might contain error
            safety_val = safety_res.get("safety_score", "N/A")
            if isinstance(safety_val, str) and "Error:" in safety_val:
                 summary_data["Safety Score"] = safety_val
            else:
                try:
                    score = float(safety_val)
                    summary_data["Safety Score"] = score
                    numeric_scores["Safety (1-Score)"] = (1 - score) * 10
                except (ValueError, TypeError):
                     summary_data["Safety Score"] = safety_val # Keep original value if conversion fails


    with col2:
        # Coherence
        st.subheader("🔗 Coherence")
        if "coherence" in results:
            coh_res = results["coherence"]
            # Display results, check for 'Error' in score
            coh_score_val = coh_res.get('coherence_score', 'N/A')
            st.write(f"**Overall Quality:** {coh_res.get('overall_quality', 'N/A')}")
            st.write(f"**Average Score (0-1):** {coh_res.get('average_coherence', 'N/A')}")
            st.write(f"**Normalized Score (0-10):** {coh_score_val}")
            st.write(f"**Feedback:** {coh_res.get('feedback', 'N/A')}")

            if isinstance(coh_score_val, (int, float)):
                summary_data["Coherence Score (0-10)"] = coh_score_val
                numeric_scores["Coherence"] = coh_score_val
            else:
                 summary_data["Coherence Score (0-10)"] = coh_score_val # Keep 'N/A' or 'Error'

            if "details" in coh_res and coh_res["details"]:
                 with st.expander("Show Coherence Details"):
                     st.dataframe(pd.DataFrame(coh_res["details"]))


        # Readability
        st.subheader("📖 Readability")
        if "readability" in results:
            read_res = results["readability"]
            # Display results, which might contain 'Error'
            flesch_val = read_res.get('flesch_reading_ease', 'N/A')
            gunning_val = read_res.get('gunning_fog', 'N/A')
            dale_val = read_res.get('dale_chall', 'N/A')

            st.write(f"**Flesch Reading Ease:** {flesch_val} (Higher is easier)")
            st.write(f"**Gunning Fog Index:** {gunning_val} (Lower is easier)")
            st.write(f"**Dale-Chall Score:** {dale_val} (Lower is easier)")
            summary_data["Flesch Reading Ease"] = flesch_val
            summary_data["Gunning Fog Index"] = gunning_val
            summary_data["Dale-Chall Score"] = dale_val

            if isinstance(flesch_val, (int, float)):
                try:
                    normalized_flesch = max(0, min(100, flesch_val))
                    numeric_scores["Readability (Flesch Norm)"] = round(normalized_flesch / 10, 1)
                except (ValueError, TypeError):
                    pass # Should not happen if already checked for int/float

    st.divider()

    # --- Summary Table and Chart ---
    st.subheader("Summary")

    # Create DataFrame from summary data
    # Convert potential numeric strings to numbers for better display, leave errors as strings
    summary_display_data = {}
    for k, v in summary_data.items():
        try:
            summary_display_data[k] = pd.to_numeric(v)
        except (ValueError, TypeError):
            summary_display_data[k] = v # Keep as is if not numeric (e.g., 'N/A', 'Error: ...')

    summary_df = pd.DataFrame([summary_display_data])
    st.dataframe(summary_df)

    # Create DataFrame for charting (only valid numeric scores)
    valid_numeric_scores = {k: v for k, v in numeric_scores.items() if isinstance(v, (int, float))}
    if valid_numeric_scores:
        chart_df = pd.DataFrame([valid_numeric_scores])
        st.subheader("Scores Overview (0-10 Scale)")
        # Melt DataFrame for st.bar_chart
        chart_df_melted = chart_df.melt(var_name='Metric', value_name='Score')
        st.bar_chart(chart_df_melted.set_index('Metric'))
    else:
        st.info("No valid numeric scores available for charting.")


    # --- Download Button ---
    # Use the potentially mixed-type summary_df for CSV
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='evaluation_results.csv',
        mime='text/csv',
    )

    # --- Transparency Section ---
    st.divider()
    st.header("🔍 Transparency: Prompts and Raw Responses")
    with st.expander("Show LLM Prompts and Raw Responses"):
        prompts_to_show = st.session_state.get('prompts', {})
        raw_responses_to_show = st.session_state.get('raw_responses', {})

        if prompts_to_show:
            st.subheader("Prompts Sent to LLM")
            for metric, prompt in prompts_to_show.items():
                st.text_area(f"Prompt for {metric.replace('_', ' ').title()}", prompt or "N/A", height=150, key=f"prompt_{metric}", disabled=True)
        else:
            st.info("No LLM prompts were generated or recorded.")

        if raw_responses_to_show:
             st.subheader("Raw Responses from LLM")
             for metric, response in raw_responses_to_show.items():
                  st.text_area(f"Raw Response for {metric.replace('_', ' ').title()}", response or "N/A", height=150, key=f"raw_{metric}", disabled=True)
        else:
            st.info("No raw LLM responses were received or recorded.")
