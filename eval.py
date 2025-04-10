import os
import json
import re
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import dotenv

# Install required packages with pip
# pip install google.genai textstat nltk rouge_score sentence_transformers scikit-learn

# Import necessary libraries for the metrics
from google import genai
from google.genai import types
import textstat
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()

# Initialize the client once during startup
CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

class STEMEvaluator:
    """Evaluates the correctness of STEM explanations using LLMs and various metrics."""
    
    def __init__(self, model_name="gemini-2.0-flash", temperature=0.2, sentence_transformer_model="all-MiniLM-L6-v2"):
        """Initialize the evaluator with specified model and configuration."""
        self.model_name = model_name
        self.temperature = temperature
        self.sentence_model = SentenceTransformer(sentence_transformer_model)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth_func = SmoothingFunction().method1

    def _get_generation_config(self, system_prompt: str, tools: Optional[List[types.Tool]] = None) -> types.GenerateContentConfig:
        """Create and return a generation configuration for the Gemini API with optional grounding tools."""
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=self.temperature,
            top_p=0.95,
            top_k=64,
            max_output_tokens=8192,
            system_instruction=system_prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "explanation_score": {
                        "type": "integer",
                        "description": "Score for the overall correctness of the explanation (1-10)"
                    },
                    "clarity_score": {
                        "type": "integer",
                        "description": "Score for how clear the explanation is (1-10)"
                    },
                    "accuracy_score": {
                        "type": "integer",
                        "description": "Score for factual accuracy of the explanation (1-10)"
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Detailed feedback on the explanation's strengths and weaknesses"
                    },
                },
                "required": ["explanation_score", "clarity_score", "accuracy_score", "feedback"]
            },
            tools=tools
        )

    def _generate_evaluation(self, prompt: str, tools: Optional[List[types.Tool]] = None) -> Dict:
        """Helper to generate text using the Gemini API, with fallback JSON extraction via regex."""
        system_prompt = (
            "You are an expert STEM education evaluator specializing in assessing explanations for correctness and clarity. "
            "Your task is to evaluate explanations by comparing them to reference materials or ground truth.\n\n"
            "Evaluation Guidelines:\n"
            "1. Correctness is the MOST important attribute - an incorrect explanation with good delivery is worse than a correct one with poor delivery\n"
            "2. Consider the subject-specific standards (mathematical precision, scientific accuracy, coding functionality)\n"
            "3. Evaluate both factual accuracy and conceptual understanding\n"
            "4. Consider clarity, structure, and accessibility of the explanation\n\n"
            "Score each category on a scale of 1-10, where:\n"
            "- 1-3: Poor (contains significant errors or misconceptions)\n"
            "- 4-6: Fair (mostly correct but with some inaccuracies)\n"
            "- 7-8: Good (correct with minor imprecisions)\n"
            "- 9-10: Excellent (completely correct, precise, and insightful)\n\n"
            "Provide specific feedback on strengths and weaknesses of the explanation."
        )
        config = self._get_generation_config(system_prompt, tools=tools)
        response = CLIENT.models.generate_content(
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

    def evaluate_with_ground_truth(self, 
                                   explanation: str, 
                                   ground_truth: str, 
                                   subject: str,
                                   question: Optional[str] = None) -> Dict:
        """Evaluate an explanation against a ground truth explanation."""
        prompt = f"""
Subject: {subject}

{f"Question: {question}" if question else ""}

Explanation to evaluate:
```
{explanation}
```

Ground truth explanation:
```
{ground_truth}
```

Evaluate the given explanation against the ground truth. Focus on correctness first, then clarity.
"""
        return self._generate_evaluation(prompt)

    def evaluate_with_rag(self, 
                          explanation: str, 
                          rag_material: str, 
                          subject: str,
                          question: Optional[str] = None) -> Dict:
        """Evaluate an explanation using RAG (Retrieval Augmented Generation) material as reference."""
        prompt = f"""
Subject: {subject}

{f"Question: {question}" if question else ""}

Explanation to evaluate:
```
{explanation}
```

Reference material:
```
{rag_material}
```

Evaluate the given explanation using the reference material. Check if the explanation's facts align with the reference material.
Focus on correctness first, then clarity.
"""
        return self._generate_evaluation(prompt)

    def evaluate_coding_solution(self, 
                                 explanation: str, 
                                 test_cases: List[Dict], 
                                 expected_output: str,
                                 actual_output: str,
                                 subject: str = "Coding") -> Dict:
        """Evaluate a coding explanation with test cases and execution results."""
        test_cases_str = json.dumps(test_cases, indent=2)
        prompt = f"""
Subject: {subject}

Coding explanation to evaluate:
```
{explanation}
```

Test Cases:
```
{test_cases_str}
```

Expected Output:
```
{expected_output}
```

Actual Output:
```
{actual_output}
```

Evaluate the coding explanation. Check if the solution correctly solves the problem and if the explanation accurately describes the approach.
The actual output should match the expected output for a correct solution.
"""
        return self._generate_evaluation(prompt)

    def evaluate_with_web_search(self, 
                                 explanation: str, 
                                 subject: str,
                                 question: Optional[str] = None) -> Dict:
        """Evaluate an explanation using grounding via web search.
           Instead of using the google_search tool (which is unsupported here), we rely solely on the response schema.
        """
        prompt = f"""
Subject: {subject}

{f"Question: {question}" if question else ""}

Explanation to evaluate:
```
{explanation}
```

Evaluate the given explanation by referencing available web search data. Focus on correctness first, then clarity.
"""
        # Do not pass tools parameter to avoid ClientError.
        return self._generate_evaluation(prompt)

    def evaluate_clarity(self, explanation: str) -> Dict:
        """Evaluate the clarity of an explanation using LLM."""
        prompt = f"""
Evaluate the clarity of the following explanation. Rate it on a scale of 1-10, 
where 1 is extremely unclear and 10 is crystal clear.

Explanation:
```
{explanation}
```

Consider:
- Is the language simple and straightforward?
- Are technical terms explained?
- Are examples provided where helpful?
- Is there a logical flow to the explanation?
"""
        
        system_prompt = "You are an expert evaluator of educational content. Provide an objective assessment of the clarity of explanations."
        
        config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            max_output_tokens=1024,
            system_instruction=system_prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "clarity_score": {
                        "type": "integer",
                        "description": "Score for clarity on a scale from 1-10"
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Brief explanation of the clarity score"
                    }
                },
                "required": ["clarity_score", "feedback"]
            },
            response_mime_type="application/json"
        )
        
        response = CLIENT.models.generate_content(
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
                return {"clarity_score": 0, "feedback": "Could not evaluate clarity."}

    def evaluate_readability(self, explanation: str) -> Dict:
        """Calculate readability score using Flesch-Kincaid Grade Level."""
        grade_level = textstat.flesch_kincaid_grade(explanation)
        
        # Normalize grade level to a 1-10 scale (lower grade level is better)
        # Grade level 20+ maps to 1, grade level 5 or below maps to 10
        if grade_level >= 20:
            score = 1
        elif grade_level <= 5:
            score = 10
        else:
            # Linear mapping from 5-20 to 10-1
            score = 10 - ((grade_level - 5) * 9 / 15)
            
        return {
            "readability_score": round(score, 2),
            "grade_level": grade_level,
            "feedback": f"The explanation requires a {grade_level:.1f} grade education level to understand."
        }

    def evaluate_coherence(self, explanation: str) -> Dict:
        """Evaluate the coherence between paragraphs in an explanation."""
        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', explanation) if p.strip()]
        
        if len(paragraphs) < 2:
            return {
                "coherence_score": 0,
                "feedback": "Cannot evaluate coherence for text with fewer than 2 paragraphs.",
                "average_coherence": 0,
                "overall_quality": "N/A"
            }
            
        # Compute embeddings
        embeddings = self.sentence_model.encode(paragraphs)
        
        # Compute coherence scores between consecutive paragraphs
        coherence_scores = []
        for i in range(len(paragraphs) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            coherence_scores.append({
                "between": (i, i+1),
                "score": round(float(sim), 4),
                "quality": self._label_coherence_quality(sim)
            })
            
        avg_score = round(np.mean([s["score"] for s in coherence_scores]), 4)
        normalized_score = round(avg_score * 10, 2)  # Convert 0-1 scale to 0-10
        
        return {
            "coherence_score": normalized_score,
            "average_coherence": avg_score,
            "overall_quality": self._label_coherence_quality(avg_score),
            "details": coherence_scores,
            "feedback": f"The explanation has {self._label_coherence_quality(avg_score).lower()} coherence between paragraphs."
        }
    
    def _label_coherence_quality(self, score):
        """Label coherence quality based on score."""
        if score > 0.85:
            return "Excellent"
        elif score > 0.65:
            return "Good"
        elif score > 0.45:
            return "Average"
        else:
            return "Poor"

    def compare_explanations(self, explanation_a: str, explanation_b: str, subject: str, question: Optional[str] = None) -> Dict:
        """Compare two explanations (A/B testing) using LLM."""
        prompt = f"""
Subject: {subject}

{f"Question: {question}" if question else ""}

Explanation A:
```
{explanation_a}
```

Explanation B:
```
{explanation_b}
```

Compare these two explanations. Which one is clearer, more accurate, and more helpful for learning?
Consider:
1. Correctness: Which explanation is more factually accurate?
2. Clarity: Which explanation is easier to understand?
3. Completeness: Which explanation covers the topic more thoroughly?
4. Pedagogy: Which explanation would be more helpful for learning?
"""
        
        system_prompt = "You are an expert evaluator of educational content. Compare explanations objectively and provide a detailed analysis."
        
        config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            max_output_tokens=2048,
            system_instruction=system_prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "winner": {
                        "type": "string",
                        "enum": ["A", "B", "Tie"],
                        "description": "Which explanation is better overall"
                    },
                    "comparison_score": {
                        "type": "integer",
                        "description": "Score representing how much better the winner is (1-10, where 1=barely better, 10=significantly better)"
                    },
                    "correctness_winner": {
                        "type": "string",
                        "enum": ["A", "B", "Tie"],
                        "description": "Which explanation is more factually accurate"
                    },
                    "clarity_winner": {
                        "type": "string",
                        "enum": ["A", "B", "Tie"],
                        "description": "Which explanation is clearer"
                    },
                    "completeness_winner": {
                        "type": "string",
                        "enum": ["A", "B", "Tie"],
                        "description": "Which explanation is more complete"
                    },
                    "pedagogy_winner": {
                        "type": "string",
                        "enum": ["A", "B", "Tie"],
                        "description": "Which explanation is better for learning"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation of the comparison results"
                    }
                },
                "required": ["winner", "comparison_score", "correctness_winner", "clarity_winner", 
                             "completeness_winner", "pedagogy_winner", "reasoning"]
            },
            response_mime_type="application/json"
        )
        
        response = CLIENT.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=config
        )
        
        try:
            result = json.loads(response.text)
            # Convert comparison_score to a 0-1 scale for the final formula
            if "comparison_score" in result:
                result["normalized_score"] = result["comparison_score"] / 10.0
            return result
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', response.text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            else:
                return {"error": "Could not compare explanations."}

    def evaluate_final_answer(self, explanation: str, correct_answer: str, subject: str) -> Dict:
        """Evaluate if the final answer in an explanation is correct."""
        prompt = f"""
Subject: {subject}

Explanation:
```
{explanation}
```

Correct answer:
```
{correct_answer}
```

Evaluate whether the explanation leads to or contains the correct answer.
Focus only on the correctness of the final result or conclusion, not the explanation process.
"""
        
        system_prompt = "You are an expert evaluator for STEM subjects. Assess whether explanations arrive at the correct answer."
        
        config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            max_output_tokens=1024,
            system_instruction=system_prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "correct": {
                        "type": "boolean",
                        "description": "Whether the explanation contains the correct answer"
                    },
                    "accuracy_score": {
                        "type": "integer",
                        "description": "How accurate the answer is (1-10)"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Explanation of the evaluation"
                    }
                },
                "required": ["correct", "accuracy_score", "explanation"]
            },
            response_mime_type="application/json"
        )
        
        response = CLIENT.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=config
        )
        
        try:
            result = json.loads(response.text)
            # Normalize score to 0-1 for final formula
            if "accuracy_score" in result:
                result["normalized_score"] = result["accuracy_score"] / 10.0
            return result
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', response.text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            else:
                return {"correct": False, "accuracy_score": 0, "explanation": "Could not evaluate final answer."}

    def evaluate_relevancy(self, explanation: str, reference: str) -> Dict:
        """Evaluate relevancy using BLEU and ROUGE scores."""
        # Prepare for BLEU calculation
        explanation_tokens = explanation.split()
        reference_tokens = reference.split()
        references = [reference_tokens]  # BLEU expects a list of references
        
        # Calculate BLEU score
        bleu_score = sentence_bleu(references, explanation_tokens, smoothing_function=self.smooth_func)
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, explanation)
        
        # Calculate average F1 score across ROUGE metrics
        rouge_f1_avg = (
            rouge_scores['rouge1'].fmeasure + 
            rouge_scores['rouge2'].fmeasure + 
            rouge_scores['rougeL'].fmeasure
        ) / 3
        
        # Combine BLEU and ROUGE into a single relevancy score (simple average)
        combined_score = (bleu_score + rouge_f1_avg) / 2
        
        # Scale to 0-10 for readability
        scaled_score = combined_score * 10
        
        return {
            "relevancy_score": round(scaled_score, 2),
            "bleu_score": round(bleu_score, 4),
            "rouge1_f1": round(rouge_scores['rouge1'].fmeasure, 4),
            "rouge2_f1": round(rouge_scores['rouge2'].fmeasure, 4),
            "rougeL_f1": round(rouge_scores['rougeL'].fmeasure, 4),
            "normalized_score": combined_score,  # 0-1 scale for final formula
            "feedback": f"The explanation has {self._label_relevancy_quality(combined_score)} relevancy to the reference."
        }
    
    def _label_relevancy_quality(self, score):
        """Label relevancy quality based on score."""
        if score > 0.8:
            return "excellent"
        elif score > 0.6:
            return "good"
        elif score > 0.4:
            return "moderate"
        elif score > 0.2:
            return "low"
        else:
            return "very low"

    def evaluate_safety(self, explanation: str) -> Dict:
        """Evaluate content safety/toxicity using Gemini."""
        prompt = f"""
Evaluate the following explanation for safety and toxicity.
Rate it on a scale of 1-10, where:
1 = highly toxic/unsafe
10 = completely safe and appropriate for educational use

Explanation:
```
{explanation}
```

Look for:
- Harmful content
- Discriminatory language
- Inappropriate examples
- Misleading information
- Age-appropriate language
"""
        
        system_prompt = "You are a content safety specialist. Evaluate educational content for safety and appropriateness."
        
        config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            max_output_tokens=1024,
            system_instruction=system_prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "safety_score": {
                        "type": "integer",
                        "description": "Safety score from 1-10 (higher is safer)"
                    },
                    "issues_found": {
                        "type": "boolean",
                        "description": "Whether any safety issues were found"
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Explanation of the safety assessment"
                    }
                },
                "required": ["safety_score", "issues_found", "feedback"]
            },
            response_mime_type="application/json"
        )
        
        response = CLIENT.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=config
        )
        
        try:
            result = json.loads(response.text)
            # Normalize score to 0-1 for final formula
            if "safety_score" in result:
                result["normalized_score"] = result["safety_score"] / 10.0
            return result
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', response.text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            else:
                return {"safety_score": 0, "issues_found": True, "feedback": "Could not evaluate safety."}

    def measure_latency(self, explanation_fn, *args, **kwargs) -> Dict:
        """Measure the latency of generating an explanation."""
        start_time = time.time()
        result = explanation_fn(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # For latency, lower is better. Convert to a 0-10 score where:
        # ≤ 1 second = 10, ≥ 10 seconds = 1, linear in between
        if execution_time <= 1:
            latency_score = 10
        elif execution_time >= 10:
            latency_score = 1
        else:
            latency_score = 10 - (execution_time - 1) * 9 / 9
            
        return {
            "latency_score": round(latency_score, 2),
            "execution_time": round(execution_time, 3),
            "result": result,
            "normalized_score": latency_score / 10.0  # 0-1 scale for final formula
        }

    def comprehensive_evaluation(self, 
                               explanation: str, 
                               reference: Optional[str] = None,
                               correct_answer: Optional[str] = None,
                               comparison_explanation: Optional[str] = None,
                               subject: str = "General",
                               question: Optional[str] = None) -> Dict:
        """
        Perform a comprehensive evaluation using multiple metrics.
        
        Final Score = 0.2⋅C + 0.1⋅R + 0.15⋅H + 0.1⋅AB + 0.25⋅FA + 0.1⋅Rel + 0.05⋅S + 0.05⋅L
        
        Where:
        C: Clarity score
        R: Readability score
        H: Coherence score
        AB: AB Comparison score
        FA: Final Answer Accuracy score
        Rel: Relevancy score
        S: Safety/Toxicity score
        L: Latency score
        """
        results = {}
        
        # Always evaluate these metrics
        clarity_result = self.evaluate_clarity(explanation)
        results["clarity"] = clarity_result
        clarity_score = clarity_result.get("clarity_score", 0) / 10.0  # Normalize to 0-1
        
        readability_result = self.evaluate_readability(explanation)
        results["readability"] = readability_result
        readability_score = readability_result.get("readability_score", 0) / 10.0  # Normalize to 0-1
        
        coherence_result = self.evaluate_coherence(explanation)
        results["coherence"] = coherence_result
        coherence_score = coherence_result.get("coherence_score", 0) / 10.0  # Normalize to 0-1
        
        safety_result = self.evaluate_safety(explanation)
        results["safety"] = safety_result
        safety_score = safety_result.get("normalized_score", 0)  # Already normalized
        
        # Optional metrics based on provided parameters
        ab_score = 0.5  # Default to neutral if no comparison
        if comparison_explanation:
            ab_result = self.compare_explanations(explanation, comparison_explanation, subject, question)
            results["ab_comparison"] = ab_result
            # If this explanation is A and it wins, or is B and B wins, score is high
            if (ab_result.get("winner") == "A") or (ab_result.get("winner") == "B"):
                ab_score = ab_result.get("normalized_score", 0.5)
            else:  # Tie
                ab_score = 0.5
        
        fa_score = 0.0  # Default to 0 if no correct answer provided
        if correct_answer:
            fa_result = self.evaluate_final_answer(explanation, correct_answer, subject)
            results["final_answer"] = fa_result
            fa_score = fa_result.get("normalized_score", 0)
        
        relevancy_score = 0.0  # Default to 0 if no reference
        if reference:
            relevancy_result = self.evaluate_relevancy(explanation, reference)
            results["relevancy"] = relevancy_result
            relevancy_score = relevancy_result.get("normalized_score", 0)
        
        # For latency, we'll use a placeholder score if not measured explicitly
        latency_score = 0.5  # Default to mid-range
        
        # Calculate final score using the formula
        final_score = (
            0.2 * clarity_score +
            0.1 * readability_score +
            0.15 * coherence_score +
            0.1 * ab_score +
            0.25 * fa_score +
            0.1 * relevancy_score +
            0.05 * safety_score +
            0.05 * latency_score
        )
        
        # Scale to 0-100 for readability
        final_score_percentage = round(final_score * 100, 2)
        
        # Add final scores to results
        results["final_score"] = final_score_percentage
        results["component_scores"] = {
            "clarity": round(clarity_score * 10, 2),
            "readability": round(readability_score * 10, 2),
            "coherence": round(coherence_score * 10, 2),
            "ab_comparison": round(ab_score * 10, 2),
            "final_answer": round(fa_score * 10, 2),
            "relevancy": round(relevancy_score * 10, 2),
            "safety": round(safety_score * 10, 2),
            "latency": round(latency_score * 10, 2)
        }
        
        return results

# Example usage
if __name__ == "__main__":
    evaluator = STEMEvaluator()
    
    # Example explanation to evaluate
    explanation = """When a ball is thrown upward, it experiences a constant acceleration due to gravity of approximately 9.8 m/s² directed downward.

This means that even as the ball is moving upward, it is constantly slowing down at a rate of 9.8 m/s² until it reaches its maximum height, at which point its velocity is momentarily zero.

After reaching its maximum height, the ball begins falling downward, and its speed increases at the same rate of 9.8 m/s².

Throughout the entire journey, the acceleration remains constant in both magnitude and direction, pointing downward toward Earth's center."""
    
    # Comprehensive evaluation with all metrics
    results = evaluator.comprehensive_evaluation(
        explanation=explanation,
        reference="Objects near Earth's surface experience a gravitational acceleration of 9.8 m/s² directed toward Earth's center. This acceleration is constant regardless of the object's mass or velocity.",
        correct_answer="A ball thrown upward experiences constant downward acceleration of 9.8 m/s².",
        subject="Physics",
        question="Describe the motion of a ball thrown upward."
    )
    
    print("Comprehensive Evaluation:")
    print(f"Final Score: {results['final_score']}/100")
    print("\nComponent Scores:")
    for metric, score in results["component_scores"].items():
        print(f"{metric}: {score}/10")
    
    # Optionally print detailed results for each metric
    # print("\nDetailed Results:")
    # print(json.dumps(results, indent=2))