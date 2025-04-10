import os
import json
import re  # new import for regex
from google import genai
from google.genai import types
from typing import Dict, List, Optional
import dotenv
dotenv.load_dotenv()

# Initialize the client once during startup
CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

class STEMEvaluator:
    """Evaluates the correctness of STEM explanations using LLMs."""
    
    def __init__(self, model_name="gemini-2.0-flash", temperature=0):
        """Initialize the evaluator with specified model and configuration."""
        self.model_name = model_name
        self.temperature = temperature

    def _get_generation_config(self, system_prompt: str, tools: Optional[List[types.Tool]] = None) -> types.GenerateContentConfig:
        """Create and return a generation configuration for the Gemini API with optional grounding tools."""
        return types.GenerateContentConfig(

            response_mime_type="application/json",
            temperature=self.temperature,
            top_p=0.97,
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

# Example usage
if __name__ == "__main__":
    evaluator = STEMEvaluator()
    
    # Example: Evaluating a math explanation with ground truth
    math_explanation = "To solve the quadratic equation x² + 5x + 6 = 0, we can factor it as (x + 2)(x + 3) = 0, giving us x = -2 or x = -3."
    math_ground_truth = "To solve x² + 5x + 6 = 0, we factor the left side to get (x + 2)(x + 3) = 0. Using the zero product property, either x + 2 = 0 or x + 3 = 0, which gives us x = -2 or x = -3."
    
    result = evaluator.evaluate_with_ground_truth(
        explanation=math_explanation,
        ground_truth=math_ground_truth,
        subject="Mathematics",
        question="Solve the quadratic equation x² + 5x + 6 = 0."
    )
    
    print("Evaluation Result:")
    print(json.dumps(result, indent=2))
    
    # Example: Using RAG material
    physics_explanation = "When a ball is thrown upward, it experiences a constant deceleration due to gravity of approximately 9.8 m/s²."
    physics_reference = "Objects near Earth's surface experience a gravitational acceleration of 9.8 m/s² directed toward Earth's center. This acceleration is constant regardless of the object's mass."
    
    rag_result = evaluator.evaluate_with_rag(
        explanation=physics_explanation,
        rag_material=physics_reference,
        subject="Physics",
        question="Describe the motion of a ball thrown upward."
    )
    
    print("\nRAG Evaluation Result:")
    print(json.dumps(rag_result, indent=2))

