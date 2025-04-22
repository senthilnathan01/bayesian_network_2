# --- api/main.py (Single Call OpenAI GPT-4o Approach) ---
import sys
print("--- DEBUG: api/main.py TOP LEVEL EXECUTION ---", flush=True)
# Print Python version and path just in case
print(f"--- DEBUG: Python Version: {sys.version} ---", flush=True)
print(f"--- DEBUG: Python Path: {sys.path} ---", flush=True)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware # Added CORS
    print("--- DEBUG: Imported FastAPI and CORS ---", flush=True)
    app = FastAPI()
    print("--- DEBUG: FastAPI app created ---", flush=True)

    # Allow CORS for frontend development (adjust origins as needed for production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Allow all origins for simplicity during dev/demo
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print("--- DEBUG: CORS middleware added ---", flush=True)


    @app.get("/api/ping")
    def ping():
        print("--- DEBUG: /api/ping called ---", flush=True)
        return {"message": "pong"}

    print("--- DEBUG: Defined /api/ping ---", flush=True)

except Exception as e:
    print(f"--- DEBUG: ERROR during setup: {e} ---", flush=True)
    # This will likely make the app fail to start, which is intended if core imports fail
    # raise e # Re-raising might prevent helpful Vercel diagnostics? Let's keep it print for now.

from pydantic import BaseModel
import os
import openai
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List
import json
import time

print("DEBUG: Starting api/main.py execution...", flush=True)
logging.info("DEBUG: Logging configured.")

# Load environment variables (for API keys)
# Assuming .env is in the root or the directory Vercel executes from for the lambda
load_dotenv()

# Set OpenAI API Key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"DEBUG: OpenAI Key loaded: {'Yes' if openai_api_key else 'NO - MISSING!'}", flush=True)

if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not found.", flush=True)
    # In a real app, you might want to raise an error here to prevent deployment without key
    # or handle it gracefully in the endpoint.

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout) # Log to stdout for Vercel
logger = logging.getLogger(__name__)

# Define Bayesian Network structure (Directed Acyclic Graph)
# Node: List of Parents
NODE_PARENTS = {
    "A1": [], "A2": [], "A3": [], "A4": [], "A5": [], "UI": [], "H": [],
    "IS3": ["A1", "A3", "A4", "H"], # Task Understanding
    "IS4": ["A2", "UI", "H"],       # Interaction Fluency
    "IS5": ["A1", "A3", "IS3"],     # Relevant Knowledge Activation
    "IS2": ["IS4", "IS3"],         # Cognitive Load
    "IS1": ["A5", "IS2", "IS3", "IS4", "IS5"], # Confidence
    "O1": ["IS1", "IS2", "IS3", "IS5"], # Predicted Success Prob
    "O2": ["IS1", "IS2", "A5"],     # Action Speed/Efficiency
    "O3": ["IS1", "IS2", "IS3"]     # Help Seeking Likelihood
}

# List all nodes
ALL_NODES = list(NODE_PARENTS.keys())
# Nodes for which LLM estimates probabilities (those with parents)
TARGET_NODES = [node for node, parents in NODE_PARENTS.items() if parents]
# Input nodes (those with no parents)
INPUT_NODES = [node for node, parents in NODE_PARENTS.items() if not parents]


# Define Node Descriptions for the prompt and UI
NODE_DESCRIPTIONS = {
    "A1": "Domain Expertise",
    "A2": "Web Literacy",
    "A3": "Task Familiarity",
    "A4": "Goal Clarity",
    "A5": "Motivation",
    "UI": "UI State (Quality/Clarity)",
    "H": "History (Relevant past interactions - positive/negative)",
    "IS1": "Confidence",
    "IS2": "Cognitive Load",
    "IS3": "Task Understanding",
    "IS4": "Interaction Fluency",
    "IS5": "Relevant Knowledge Activation",
    "O1": "Predicted Success Probability",
    "O2": "Action Speed/Efficiency",
    "O3": "Help Seeking Likelihood"
}


class ContinuousUserInput(BaseModel):
    A1: float
    A2: float
    A3: float
    A4: float
    A5: float
    UI: float
    H: float

def call_openai_for_full_bn(input_states: Dict[str, float]) -> Dict[str, float]:
    """
    Calls OpenAI API once to estimate probabilities for all target nodes.
    Returns estimated probabilities for TARGET_NODES.
    """
    if not openai_api_key:
        logger.error("OpenAI API Key not configured during call.")
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    openai.api_key = openai_api_key # Ensure key is set for the call

    input_descriptions = []
    for node, value in input_states.items():
        # Describe the input value qualitatively
        state_desc = "High" if value >= 0.66 else ("Medium" if value >= 0.33 else "Low")
        input_descriptions.append(f"- {node} ({NODE_DESCRIPTIONS.get(node, node)}): {state_desc} (input probability approx {value:.2f})")
    input_context = "\n".join(input_descriptions)

    # Generate qualitative relationship descriptions based on the DAG structure
    relationship_descriptions = []
    for node, parents in NODE_PARENTS.items():
        if parents: # Only describe nodes with parents
            parent_names = [f"{p} ({NODE_DESCRIPTIONS.get(p, p)})" for p in parents]
            node_name = f"{node} ({NODE_DESCRIPTIONS.get(node, node)})"
            relationship_descriptions.append(f"- {node_name} is influenced by: {', '.join(parent_names)}.")
            # Add qualitative influence rules based on the new structure
            if node == "IS3": relationship_descriptions.append("  Qualitative: Higher A1, A3, A4, H typically increase IS3 (Task Understanding).")
            if node == "IS4": relationship_descriptions.append("  Qualitative: Higher A2, UI, H typically increase IS4 (Interaction Fluency).")
            if node == "IS5": relationship_descriptions.append("  Qualitative: Higher A1, A3, IS3 typically increase IS5 (Relevant Knowledge Activation).")
            if node == "IS2": relationship_descriptions.append("  Qualitative: Lower IS4, IS3 typically increase IS2 (Cognitive Load).")
            if node == "IS1": relationship_descriptions.append("  Qualitative: Higher A5, IS4, IS5 typically increase IS1 (Confidence). Lower IS2, IS3 typically decrease IS1.")
            if node == "O1": relationship_descriptions.append("  Qualitative: Higher IS1, IS4, IS5 typically increase O1 (Predicted Success Probability). Higher IS2, IS3 typically decrease O1.")
            if node == "O2": relationship_descriptions.append("  Qualitative: Higher IS1, A5 typically increase O2 (Action Speed/Efficiency). Higher IS2 typically decrease O2.")
            if node == "O3": relationship_descriptions.append("  Qualitative: Higher IS2, IS3 typically increase O3 (Help Seeking Likelihood). Higher IS1 typically decrease O3.")


    structure_description = f"""
    This Bayesian Network models user cognitive factors and actions during a web task.
    Nodes are binary (State 0 or 1). We are estimating P(Node=1).
    Node Descriptions: {json.dumps(NODE_DESCRIPTIONS, indent=2)}

    Dependencies (Node -> Parents): {json.dumps(NODE_PARENTS, indent=2)}

    Qualitative Relationship Descriptions:
    {'\n'.join(relationship_descriptions)}
    """

    system_message = """
    You are an expert probabilistic reasoner simulating a Directed Acyclic Graph (DAG) based on qualitative descriptions.
    Your task is to estimate the probability of the target nodes being in a 'High' state (equivalent to state 1), given the initial input probabilities (also representing P=1) for the input nodes.
    Use the provided DAG structure and qualitative relationship descriptions to guide your estimation process. Propagate the influence from input nodes through the intermediate nodes to the output nodes according to the described dependencies and qualitative rules.
    Provide the final estimated probabilities ONLY as a single, valid JSON object mapping the target node names to their estimated P(Node=1) value (a float between 0.0 and 1.0).
    Ensure the JSON object contains *all* target nodes: """ + ', '.join(TARGET_NODES) + """
    Example format: {"IS1": 0.75, "IS2": 0.3, ... , "O3": 0.65}
    Output ONLY the JSON object and nothing else.
    """
    user_message = f"""
    Initial Input Probabilities (P=1):
    {input_context}

    Bayesian Network Structure and Qualitative Relationships:
    {structure_description}

    Estimate the probability P(Node=1) for all target nodes ({', '.join(TARGET_NODES)}) based on the provided inputs, network structure, and qualitative relationships. Return the result ONLY as a JSON object mapping each target node name to its estimated probability (a float between 0.0 and 1.0).
    """

    logger.debug("Constructing single call prompt to OpenAI...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o", # Using gpt-4o as requested
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"}, # Requesting JSON object
            max_tokens=1000, # Increased max_tokens slightly
            temperature=0.1, # Lower temperature for less randomness
            n=1
        )
        llm_output_raw = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Raw Output (Single Call): {llm_output_raw}")

        try:
            estimated_probs = json.loads(llm_output_raw)
            validated_probs = {}
            missing_nodes = []
            # Validate and clamp probabilities, ensure all TARGET_NODES are present
            for node in TARGET_NODES:
                if node in estimated_probs and isinstance(estimated_probs[node], (float, int)):
                    validated_probs[node] = max(0.0, min(1.0, float(estimated_probs[node])))
                else:
                    logger.warning(f"Node '{node}' missing or invalid value in LLM JSON output. Received: {estimated_probs.get(node)}")
                    missing_nodes.append(node)
                    # Provide a default or handle this based on requirements
                    # Defaulting to input value if it exists, else 0.5, or better, raise error?
                    # Let's default to 0.5 and log a warning.
                    validated_probs[node] = 0.5 # Default if LLM misses a node


            if missing_nodes:
                logger.warning(f"LLM output was missing or invalid for nodes: {', '.join(missing_nodes)}. Using default 0.5 for these.")
            else:
                 logger.info("LLM output contains all expected target nodes.")

            return validated_probs

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse LLM output as JSON: {llm_output_raw}. Error: {json_err}")
            # Provide the raw output in the error detail for debugging
            raise HTTPException(status_code=500, detail=f"Failed to parse LLM JSON response. Output: {llm_output_raw}. Error: {json_err}")
        except Exception as e:
             logger.error(f"Error validating/processing LLM JSON: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Error processing LLM JSON response: {e}")


    except openai.APIError as e:
        logger.error(f"OpenAI API returned an API Error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API Error: {e.body}") # Include error body
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI Authentication Error: {e}")
        raise HTTPException(status_code=401, detail=f"OpenAI Authentication Failed - Check API Key. Error: {e.body}")
    except openai.RateLimitError as e:
        logger.error(f"OpenAI Rate Limit Exceeded: {e}")
        raise HTTPException(status_code=429, detail=f"OpenAI Rate Limit Exceeded: {e.body}")
    except Exception as e:
        logger.error(f"Error in single call OpenAI interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during OpenAI request: {e}")

@app.post("/api/predict_openai_bn_single_call")
async def predict_openai_bn_single_call(data: ContinuousUserInput):
    print("DEBUG: Entered /api/predict_openai_bn_single_call function.", flush=True)
    """
    Receives input probabilities and returns all node probabilities
    estimated by a single call to the OpenAI LLM.
    """
    # Get qualitative descriptions and structure to return to frontend
    input_descriptions_list = []
    input_probs_dict = data.dict()
    for node, value in input_probs_dict.items():
        state_desc = "High" if value >= 0.66 else ("Medium" if value >= 0.33 else "Low")
        input_descriptions_list.append({
            "node": node,
            "description": NODE_DESCRIPTIONS.get(node, node),
            "value": value,
            "state": state_desc
        })

    relationship_descriptions_list = []
    for node, parents in NODE_PARENTS.items():
         if parents:
             parent_names = [f"{p} ({NODE_DESCRIPTIONS.get(p, p)})" for p in parents]
             node_name_desc = f"{node} ({NODE_DESCRIPTIONS.get(node, node)})"
             description = f"{node_name_desc} is influenced by: {', '.join(parent_names)}."
             # Add qualitative influence rules based on the new structure
             qualitative = ""
             if node == "IS3": qualitative = "Higher A1, A3, A4, H typically increase Task Understanding."
             if node == "IS4": qualitative = "Higher A2, UI, H typically increase Interaction Fluency."
             if node == "IS5": qualitative = "Higher A1, A3, IS3 typically increase Relevant Knowledge Activation."
             if node == "IS2": qualitative = "Lower IS4, IS3 typically increase Cognitive Load."
             if node == "IS1": qualitative = "Higher A5, IS4, IS5 typically increase Confidence. Lower IS2, IS3 typically decrease Confidence."
             if node == "O1": qualitative = "Higher IS1, IS4, IS5 typically increase Predicted Success Probability. Higher IS2, IS3 typically decrease Predicted Success Probability."
             if node == "O2": qualitative = "Higher IS1, A5 typically increase Action Speed/Efficiency. Higher IS2 typically decrease Action Speed/Efficiency."
             if node == "O3": qualitative = "Higher IS2, IS3 typically increase Help Seeking Likelihood. Higher IS1 typically decrease Help Seeking Likelihood."
             relationship_descriptions_list.append({
                 "node": node,
                 "description": description,
                 "qualitative": qualitative
             })


    try:
        # Call LLM for target node probabilities
        estimated_target_probs = call_openai_for_full_bn(input_probs_dict)

        # Combine input probabilities and estimated probabilities
        all_current_probabilities = {**input_probs_dict, **estimated_target_probs}

        final_result = {}
        for node in ALL_NODES:
            if node in all_current_probabilities:
                p1 = all_current_probabilities[node]
                p1_clamped = max(0.0, min(1.0, p1))
                final_result[node] = {"0": 1.0 - p1_clamped, "1": p1_clamped, "description": NODE_DESCRIPTIONS.get(node, node)}
            else:
                # Should not happen if all nodes are in ALL_NODES and handled
                logger.warning(f"Node {node} was not found in combined probability dictionary.")
                final_result[node] = {"0": 0.5, "1": 0.5, "description": NODE_DESCRIPTIONS.get(node, node)}


        logger.info("Successfully generated prediction using single LLM call.")

        # Return probabilities AND the context provided to the LLM for display
        return {
            "probabilities": final_result,
            "llm_context": {
                "input_states": input_descriptions_list,
                "node_dependencies": NODE_PARENTS,
                "qualitative_rules": relationship_descriptions_list,
                "node_descriptions": NODE_DESCRIPTIONS
            }
        }

    except HTTPException as e:
        # Pass HTTPException errors directly (like API key missing, LLM errors)
        raise e
    except Exception as e:
        logger.error(f"Error in single call endpoint logic: {e}", exc_info=True)
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error during single call prediction: {e}")

print("DEBUG: /api/predict_openai_bn_single_call route defined.", flush=True)

@app.get("/")
def root():
    """ Basic endpoint to check if the API is running. """
    # This route is actually served by Vercel's static server from public/index.html
    # This function will likely never be called in a standard Vercel deployment
    # but it's good practice for local testing with `uvicorn api.main:app --reload`
    print("DEBUG: Entered / route (API function).", flush=True)
    return {"message": "API is running. Frontend should be served from /public."}

print("DEBUG: Finished defining routes.", flush=True)
