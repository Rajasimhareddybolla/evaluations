# Fix for PyTorch/Streamlit compatibility issue - Must be at the very top of the file
import os
import sys

# Configure Streamlit to exclude torch modules from module watching
# Specifically target torch.classes which causes the error with __path__._path
os.environ["STREAMLIT_WATCH_EXCLUDES"] = "torch,torchvision,torchaudio,torch._classes"
os.environ["STREAMLIT_SERVER_WATCH_EXCLUDES"] = "torch,torchvision,torchaudio,torch._classes"
import json
import re
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import dotenv
from concurrent.futures import ThreadPoolExecutor
import threading

# Now it's safe to import torch after setting the environment variables
import torch

# Import necessary libraries
from google import genai
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables
dotenv.load_dotenv()

# Initialize clients
login(token=os.environ["hf_token"])

class ImprovedSTEMEvaluator:
    """
    Enhanced evaluator for STEM explanations focusing on 6 key metrics:
    1. Accuracy
    2. Completeness
    3. Coherence
    4. Readability
    5. Clarity
    6. Safety/Toxicity
    """
    
    def __init__(self,api_key ,  model_name="gemini-2.0-flash-lite", temperature=0.1, 
                 sentence_transformer_model="all-MiniLM-L6-v2"):
        """Initialize the evaluator with specified models and configurations."""
        self.model_name = model_name
        self.temperature = temperature
        self.sentence_model = SentenceTransformer(sentence_transformer_model)
        self.client = genai.Client(api_key=api_key)
        
        # Initialize toxicity detection model
        try:
            self.toxicity_model = pipeline(
                "text-classification", 
                model="microsoft/MiniLM-L12-H384-uncased-detoxify"
            )
        except:
            print("Warning: Could not load toxicity model. Will use fallback method.")
            self.toxicity_model = None

    def _create_gemini_response_schema(self, metric_name):
        """Create a response schema for structured Gemini evaluations."""
        schema = {
            "type": "object",
            "properties": {
                f"{metric_name}_score": {
                    "type": "integer",
                    "description": f"Score for {metric_name} on a scale from 1-10"
                },
                "analysis": {
                    "type": "string",
                    "description": f"Detailed analysis of the {metric_name}"
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Key strengths related to {metric_name}"
                },
                "weaknesses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Areas for improvement related to {metric_name}"
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific suggestions for improvement"
                }
            },
            "required": [f"{metric_name}_score", "analysis", "strengths", "weaknesses", "suggestions"]
        }
        return schema
    
    def _generate_structured_evaluation(self, prompt, system_prompt, schema):
        """Helper to generate structured evaluations using Gemini API."""
        from google.genai import types
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=self.temperature,
            top_p=0.95,
            top_k=64,
            max_output_tokens=4096,
            system_instruction=system_prompt,
            response_schema=schema
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=config
        )
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', response.text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            else:
                return {"error": "Could not extract JSON from response."}

    def evaluate_accuracy(self, 
                         explanation: str, 
                         reference: Optional[str] = None,
                         question: Optional[str] = None,
                         subject: str = "General STEM") -> Dict:
        """
        Evaluate the factual accuracy of an explanation.
        
        Args:
            explanation: The explanation to evaluate
            reference: Optional reference material or ground truth explanation
            question: Optional question that the explanation addresses
            subject: Subject area of the explanation
            
        Returns:
            Dict containing accuracy score and detailed analysis
        """
        system_prompt = """
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
        """
        
        prompt = f"""
        Subject: {subject}
        
        {f"Question: {question}" if question else ""}
        
        Explanation to evaluate:
        ```
        {explanation}
        ```
        
        {f"Reference material:\n```\n{reference}\n```" if reference else "Evaluate based on standard STEM knowledge in this field."}
        
        Perform a comprehensive accuracy evaluation of this explanation. Identify any factual errors, 
        misconceptions, or imprecisions. Check for correctness of concepts, equations, calculations, 
        and scientific/mathematical principles.
        """
        
        schema = self._create_gemini_response_schema("accuracy")
        return self._generate_structured_evaluation(prompt, system_prompt, schema)

    def evaluate_completeness(self, 
                             explanation: str, 
                             reference: Optional[str] = None,
                             question: Optional[str] = None,
                             subject: str = "General STEM") -> Dict:
        """
        Evaluate the completeness of an explanation.
        
        Args:
            explanation: The explanation to evaluate
            reference: Optional reference material with complete information
            question: Optional question that the explanation addresses
            subject: Subject area of the explanation
            
        Returns:
            Dict containing completeness score and detailed analysis
        """
        system_prompt = """
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
        """
        
        prompt = f"""
        Subject: {subject}
        
        {f"Question: {question}" if question else ""}
        
        Explanation to evaluate:
        ```
        {explanation}
        ```
        
        {f"Reference material for complete coverage:\n```\n{reference}\n```" if reference else ""}
        
        Evaluate how complete this explanation is. Identify any missing key concepts, steps, applications,
        or components that should be included for a comprehensive understanding of the topic.
        """
        
        schema = self._create_gemini_response_schema("completeness")
        return self._generate_structured_evaluation(prompt, system_prompt, schema)

    def evaluate_coherence(self, explanation: str) -> Dict:
        """
        Evaluate the coherence of an explanation using both ML techniques and LLM assessment.
        
        Args:
            explanation: The explanation to evaluate
            
        Returns:
            Dict containing coherence score and detailed analysis
        """
        # First use embedding-based coherence measurement
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', explanation) if p.strip()]
        
        ml_coherence_score = 0
        paragraph_transitions = []
        
        if len(paragraphs) >= 2:
            # Compute embeddings
            embeddings = self.sentence_model.encode(paragraphs)
            
            # Compute coherence scores between consecutive paragraphs
            scores = []
            for i in range(len(paragraphs) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                scores.append(sim)
                paragraph_transitions.append({
                    "paragraphs": (i+1, i+2),
                    "score": round(float(sim), 4),
                })
                
            ml_coherence_score = np.mean(scores)
        
        # Then use Gemini for qualitative assessment
        system_prompt = """
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
        """
        
        prompt = f"""
        Explanation to evaluate:
        ```
        {explanation}
        ```
        
        Evaluate the coherence and logical flow of this explanation. Consider how well ideas connect,
        whether concepts build logically on each other, and if the explanation maintains a clear
        thread throughout. Identify any issues with transitions, contradictions, or logical jumps.
        
        Additional ML-based coherence analysis:
        - ML-calculated coherence score between paragraphs: {round(ml_coherence_score, 4) if len(paragraphs) >= 2 else "N/A"}
        - Number of paragraphs: {len(paragraphs)}
        """
        
        schema = self._create_gemini_response_schema("coherence")
        llm_result = self._generate_structured_evaluation(prompt, system_prompt, schema)
        
        # Combine ML and LLM results
        result = llm_result
        result["ml_coherence_score"] = round(float(ml_coherence_score), 4) if len(paragraphs) >= 2 else 0
        result["paragraph_transitions"] = paragraph_transitions
        
        # Weighted average: 70% LLM assessment, 30% ML-based measurement (if available)
        if "coherence_score" in result and len(paragraphs) >= 2:
            llm_score = result["coherence_score"]
            # Convert ML score from 0-1 to 1-10 scale
            ml_score_scaled = 1 + (ml_coherence_score * 9)
            result["final_coherence_score"] = round(0.7 * llm_score + 0.3 * ml_score_scaled, 1)
        else:
            result["final_coherence_score"] = result.get("coherence_score", 0)
            
        return result

    def evaluate_readability(self, explanation: str, target_audience: str = "general") -> Dict:
        """
        Evaluate the readability of an explanation for LLM consumers.
        
        Args:
            explanation: The explanation to evaluate
            target_audience: Target audience (e.g., "beginner", "advanced", "general")
            
        Returns:
            Dict containing readability score and detailed analysis
        """
        # Calculate standard readability metrics
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(explanation)
        flesch_reading_ease = textstat.flesch_reading_ease(explanation)
        smog_index = textstat.smog_index(explanation)
        
        # Analyze text statistics
        avg_sentence_length = len(re.findall(r'[.!?]+', explanation)) / max(1, len(explanation.split()))
        avg_word_length = sum(len(word) for word in explanation.split()) / max(1, len(explanation.split()))
        
        system_prompt = """
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
        """
        
        prompt = f"""
        Explanation to evaluate:
        ```
        {explanation}
        ```
        
        Target audience: {target_audience}
        
        Evaluate how readable this explanation is for LLM users of the specified target audience.
        Focus on appropriate complexity, clarity of technical terminology, and overall accessibility.
        
        Text statistics analysis:
        - Flesch-Kincaid Grade Level: {flesch_kincaid_grade} (lower is more readable)
        - Flesch Reading Ease: {flesch_reading_ease} (higher is more readable)
        - SMOG Index: {smog_index}
        - Average sentence length: {avg_sentence_length:.2f}
        - Average word length: {avg_word_length:.2f} characters
        
        Note: For LLM users, standard readability metrics should be considered but not followed rigidly.
        Focus on appropriate complexity for the subject and audience, clarity of explanation, and
        effective communication over strict adherence to readability formulas.
        """
        
        schema = self._create_gemini_response_schema("readability")
        result = self._generate_structured_evaluation(prompt, system_prompt, schema)
        
        # Add the computed metrics to the result
        result["metrics"] = {
            "flesch_kincaid_grade": round(flesch_kincaid_grade, 2),
            "flesch_reading_ease": round(flesch_reading_ease, 2),
            "smog_index": round(smog_index, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": round(avg_word_length, 2)
        }
        
        return result

    def evaluate_clarity(self, 
                        explanation: str, 
                        question: Optional[str] = None,
                        subject: str = "General STEM") -> Dict:
        """
        Evaluate the clarity of an explanation.
        
        Args:
            explanation: The explanation to evaluate
            question: Optional question that the explanation addresses
            subject: Subject area of the explanation
            
        Returns:
            Dict containing clarity score and detailed analysis
        """
        system_prompt = """
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
        """
        
        prompt = f"""
        Subject: {subject}
        
        {f"Question: {question}" if question else ""}
        
        Explanation to evaluate:
        ```
        {explanation}
        ```
        
        Evaluate how clear this explanation is. Consider the structure, organization, use of examples,
        and effectiveness of communication. Identify specific aspects that enhance or diminish clarity.
        """
        
        schema = self._create_gemini_response_schema("clarity")
        return self._generate_structured_evaluation(prompt, system_prompt, schema)

    def evaluate_safety(self, explanation: str) -> Dict:
        """
        Evaluate the safety/toxicity of content using specialized models.
        
        Args:
            explanation: The explanation to evaluate
            
        Returns:
            Dict containing safety score and detailed analysis
        """
        # Initialize results
        safety_score = 10  # Assume safe by default
        flags = []
        toxicity_score = 0
        
        # Check for potentially harmful content using toxicity model if available
        if self.toxicity_model:
            try:
                result = self.toxicity_model(explanation)
                if isinstance(result, list) and len(result) > 0:
                    toxicity_score = result[0].get('score', 0)
                    
                    # Convert toxicity score (0-1) to safety score (1-10, inverted)
                    safety_score = max(1, min(10, round(10 - (toxicity_score * 9))))
                    
                    if toxicity_score > 0.5:
                        flags.append("High toxicity detected")
                    elif toxicity_score > 0.3:
                        flags.append("Moderate toxicity detected")
            except Exception as e:
                print(f"Error in toxicity model: {str(e)}")
        
        # Check for specific STEM safety concerns using Gemini
        system_prompt = """
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
        
        prompt = f"""
        Explanation to evaluate:
        ```
        {explanation}
        ```
        
        Evaluate this explanation for any safety concerns, harmful content, misinformation,
        bias, or ethically problematic elements. Focus specifically on STEM-related safety issues.
        
        If you find any concerns, please be specific about what is problematic and why.
        If the content appears safe and appropriate, state that as well.
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 1024*4,
                "system_instruction":system_prompt
            },
        )
        
        analysis = response.text
        safety_concerns = []
        
        # Parse the response for safety concerns
        if "no safety concerns" not in analysis.lower() and "appears safe" not in analysis.lower():
            # Extract specific concerns if possible
            concern_lines = [line.strip() for line in analysis.split('\n') if line.strip() 
                            and not line.strip().startswith("The explanation") 
                            and not line.strip().startswith("Overall")]
            
            if concern_lines:
                safety_concerns = concern_lines
                # Adjust safety score based on concerns detected
                adjustment = min(len(concern_lines) * 2, 6)  # Maximum adjustment of 6 points
                safety_score = max(1, safety_score - adjustment)
                
        # Construct result
        result = {
            "safety_score": safety_score,
            "normalized_score": safety_score / 10.0,
            "toxicity_score": round(toxicity_score, 4) if self.toxicity_model else "N/A",
            "safety_concerns": safety_concerns,
            "flags": flags,
            "analysis": analysis,
            "is_safe": safety_score >= 7
        }
        
        return result

    def comprehensive_evaluation(self, 
                               explanation: str, 
                               reference: Optional[str] = None,
                               question: Optional[str] = None,
                               subject: str = "General STEM",
                               target_audience: str = "general") -> Dict:
        """
        Perform a comprehensive evaluation using all six metrics.
        
        Args:
            explanation: The explanation to evaluate
            reference: Optional reference material or ground truth
            question: Optional question that the explanation addresses
            subject: Subject area of the explanation
            target_audience: Target audience for readability assessment
            
        Returns:
            Dict containing scores for all metrics and overall evaluation
        """
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
            futures = [executor.submit(run_evaluation, name, func, *args) for name, func, *args in tasks]
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
            "summary": self._generate_evaluation_summary(explanation, scores, weighted_score, question, subject)
        }
        
        return comprehensive_result
        
    def _generate_evaluation_summary(self, explanation, scores, overall_score, question, subject):
        """Generate a concise summary of the evaluation results."""
        metrics_summary = ", ".join([f"{metric}: {score}/10" for metric, score in scores.items()])
        
        system_prompt = """
        You are an expert STEM content evaluator providing concise, professional summaries of content quality.
        Your summary should highlight key strengths and weaknesses based on the evaluation metrics.
        Keep your response brief, objective, and actionable.
        """
        
        prompt = f"""
        Subject: {subject}
        {f"Question: {question}" if question else ""}
        
        Metrics Scores (out of 10):
        {metrics_summary}
        
        Overall Score: {round(overall_score, 2)}/10
        
        Based on these evaluation metrics, provide a brief, professional summary of the explanation's 
        quality. Highlight 2-3 key strengths and 2-3 areas for improvement. Keep your response 
        under 150 words and focus on the most important aspects.
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 3000,
                "system_instruction":system_prompt
            },
        )
        
        return response.text

# Example usage
if __name__ == "__main__":
    evaluator = ImprovedSTEMEvaluator()
    
    # Example explanation to evaluate
    explanation = """
    The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse 
    equals the sum of the squares of the lengths of the other two sides. If we denote the sides as a, b, 
    and c (where c is the hypotenuse), the theorem is expressed as: a² + b² = c².
    
    This fundamental relationship has numerous applications in mathematics, physics, engineering, and 
    everyday life. For example, it allows us to calculate distances and ensure structural stability in 
    construction.
    
    We can prove the theorem in several ways. One approach uses similar triangles, while another uses 
    area calculations. The theorem works only for right triangles but generalizes to other relationships 
    like the law of cosines for non-right triangles.
    """
    
    # Comprehensive evaluation
    results = evaluator.comprehensive_evaluation(
        explanation=explanation,
        reference="The Pythagorean theorem relates the sides of a right triangle: a² + b² = c², where c is the hypotenuse.",
        question="Explain the Pythagorean theorem and its significance.",
        subject="Mathematics",
        target_audience="high school"
    )
    
    print(f"Overall Evaluation Score: {results['overall_percentage']}%")
    print("\nIndividual Scores:")
    for metric, score in results['scores'].items():
        print(f"{metric.capitalize()}: {score}/10")
    
    print("\nSummary:")
    print(results['summary'])
