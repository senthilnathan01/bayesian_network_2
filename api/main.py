import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4
import urllib.parse # Needed to parse Redis URL

print("--- DEBUG: api/main.py TOP LEVEL EXECUTION ---", flush=True)
print(f"--- DEBUG: Python Version: {sys.version} ---", flush=True)
print(f"--- DEBUG: Python Path: {sys.path} ---", flush=True)

try:
    from fastapi import FastAPI, HTTPException, Body
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import openai
    from dotenv import load_dotenv
    import redis # Import standard Redis client
    print("--- DEBUG: Imported FastAPI, CORS, Pydantic, OpenAI, Redis ---", flush=True)

    app = FastAPI()
    print("--- DEBUG: FastAPI app created ---", flush=True)

    # Allow CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print("--- DEBUG: CORS middleware added ---", flush=True)

except Exception as e:
    print(f"--- DEBUG: ERROR during core imports: {e} ---", flush=True)
    raise e

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# Get Redis connection details from environment variables set by Vercel linking Upstash
# Vercel commonly provides UPSTASH_REDIS_URL or REDIS_URL
redis_url = os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL")

redis_client = None
if redis_url:
    try:
        # Use from_url to handle connection details including password and SSL
        redis_client = redis.from_url(redis_url, decode_responses=True) # decode_responses=True to get strings back
        # Test connection
        redis_client.ping()
        logger.info("--- Successfully connected to Redis (Upstash) ---")
    except Exception as e:
        logger.error(f"--- Failed to connect to Redis using URL: {e} ---", exc_info=True)
        redis_client = None # Ensure client is None if connection fails
else:
    logger.warning("--- UPSTASH_REDIS_URL or REDIS_URL environment variable not found. Redis features disabled. ---")


if not openai_api_key:
    logger.warning("OPENAI_API_KEY environment variable not found.")

# --- Pydantic Models (Unchanged) ---

class NodeData(BaseModel):
    id: str
    fullName: str = Field(..., description="User-friendly name for the node")
    nodeType: str = Field(..., description="Type of node: 'input', 'hidden'")

class EdgeData(BaseModel):
    source: str
    target: str

class GraphStructure(BaseModel):
    nodes: List[NodeData]
    edges: List[EdgeData]

class ContinuousUserInput(BaseModel):
    input_values: Dict[str, float]

class PredictionPayload(ContinuousUserInput):
    graph_structure: GraphStructure

class SaveConfigPayload(BaseModel):
    config_name: str = Field(..., description="User-provided name for the configuration")
    graph_structure: GraphStructure

# --- Helper Functions (Unchanged, except they don't interact with DB) ---

def get_dynamic_node_info(graph: GraphStructure):
    """ Parses graph structure to get node info needed for LLM prompt """
    node_parents = {node.id: [] for node in graph.nodes}
    node_descriptions = {node.id: node.fullName for node in graph.nodes}
    node_types = {node.id: node.nodeType for node in graph.nodes}

    for edge in graph.edges:
        if edge.target in node_parents:
            node_parents[edge.target].append(edge.source)
        else:
             logger.warning(f"Edge target '{edge.target}' not found in node list.")

    all_nodes = list(node_parents.keys())
    target_nodes = [node_id for node_id, parents in node_parents.items() if node_types.get(node_id) == 'hidden'] # Only hidden nodes need prediction
    input_nodes = [node_id for node_id, node_type in node_types.items() if node_type == 'input']

    relationship_descriptions = []
    for node_id in target_nodes:
        parents = node_parents.get(node_id, [])
        if parents:
            parent_names = [f"{p} ({node_descriptions.get(p, p)})" for p in parents]
            node_name = f"{node_id} ({node_descriptions.get(node_id, node_id)})"
            relationship_descriptions.append(f"- {node_name} is influenced by: {', '.join(parent_names)}.")
            relationship_descriptions.append(f"  Qualitative: Generally, higher values in parent nodes tend to increase {node_name}, but complex interactions exist.")

    return node_parents, node_descriptions, node_types, all_nodes, target_nodes, input_nodes, relationship_descriptions


def call_openai_dynamic_bn(
    input_states: Dict[str, float],
    graph: GraphStructure
) -> Dict[str, float]:
    """ Calls OpenAI API for dynamically defined BN (Unchanged Logic) """
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    openai.api_key = openai_api_key

    node_parents, node_descriptions, _, _, target_nodes, input_nodes, relationship_descriptions = get_dynamic_node_info(graph)

    missing_inputs = [node for node in input_nodes if node not in input_states]
    if missing_inputs:
         raise HTTPException(status_code=400, detail=f"Missing input values for nodes: {', '.join(missing_inputs)}")

    input_desc_list = []
    for node, value in input_states.items():
        if node in input_nodes:
            state_desc = "High" if value >= 0.66 else ("Medium" if value >= 0.33 else "Low")
            input_desc_list.append(f"- {node} ({node_descriptions.get(node, node)}): {state_desc} (input probability approx {value:.2f})")
    input_context = "\n".join(input_desc_list)

    if not target_nodes:
        logger.info("No target (hidden) nodes found in the graph. Returning empty results.")
        return {}

    #--(Prompt generation logic unchanged)--
    structure_description = f"""
    This Bayesian Network models user cognitive factors and actions during a web task, defined by the user.
    Nodes are binary (State 0 or 1). We are estimating P(Node=1).
    Node Descriptions: {json.dumps(node_descriptions, indent=2)}
    Node Types: {json.dumps({node.id: node.nodeType for node in graph.nodes}, indent=2)}
    Dependencies (Node <- Parents): {json.dumps(node_parents, indent=2)}
    Qualitative Relationship Descriptions:
    {'\n'.join(relationship_descriptions)}
    """
    system_message = f"""
    You are an expert probabilistic reasoner simulating a user-defined Directed Acyclic Graph (DAG).
    Your task is to estimate the probability P(Node=1) for the 'hidden' type nodes, given the initial input probabilities P(Node=1) for the 'input' type nodes.
    Use the provided DAG structure and qualitative relationship descriptions to guide your estimation.
    Provide the final estimated probabilities ONLY as a single, valid JSON object mapping the target node names ({', '.join(target_nodes)}) to their estimated P(Node=1) value (a float between 0.0 and 1.0).
    Ensure the JSON object contains *all* target nodes. Example format: {{"HiddenNode1": 0.7, "HiddenNode2": 0.4}}
    Output ONLY the JSON object.
    """
    user_message = f"""
    Initial Input Probabilities (P=1):
    {input_context}
    Bayesian Network Structure and Qualitative Relationships:
    {structure_description}
    Estimate the probability P(Node=1) for all target nodes ({', '.join(target_nodes)}) based on the provided inputs, network structure, and qualitative relationships. Return the result ONLY as a JSON object mapping each target node name to its estimated probability (a float between 0.0 and 1.0).
    """

    logger.debug("Constructing dynamic prompt for OpenAI...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[ {"role": "system", "content": system_message}, {"role": "user", "content": user_message} ],
            response_format={"type": "json_object"},
            max_tokens=1500, temperature=0.1, n=1
        )
        llm_output_raw = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Raw Output (Dynamic Call): {llm_output_raw}")

        # -- (JSON Parsing and Validation Logic Unchanged) --
        try:
            estimated_probs = json.loads(llm_output_raw)
            validated_probs = {}
            missing_nodes = []
            for node in target_nodes:
                if node in estimated_probs and isinstance(estimated_probs[node], (float, int)):
                    validated_probs[node] = max(0.0, min(1.0, float(estimated_probs[node])))
                else:
                    logger.warning(f"Node '{node}' missing/invalid in LLM JSON. Using default 0.5.")
                    missing_nodes.append(node)
                    validated_probs[node] = 0.5
            if missing_nodes: logger.warning(f"LLM output missing/invalid for: {', '.join(missing_nodes)}.")
            return validated_probs
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse LLM output as JSON: {llm_output_raw}. Error: {json_err}")
            raise HTTPException(status_code=500, detail=f"Failed to parse LLM JSON response. Output: {llm_output_raw}. Error: {json_err}")
        except Exception as e:
             logger.error(f"Error validating/processing LLM JSON: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Error processing LLM JSON response: {e}")

    # -- (OpenAI Error Handling Unchanged) --
    except openai.APIError as e: raise HTTPException(status_code=502, detail=f"OpenAI API Error: {e.body}")
    except openai.AuthenticationError as e: raise HTTPException(status_code=401, detail=f"OpenAI Authentication Failed. Error: {e.body}")
    except openai.RateLimitError as e: raise HTTPException(status_code=429, detail=f"OpenAI Rate Limit Exceeded: {e.body}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error during OpenAI request: {e}")


# --- API Endpoints ---

@app.get("/api/ping")
def ping():
    logger.info("--- /api/ping called ---")
    if redis_client:
        try:
            redis_client.ping()
            return {"message": "pong", "redis_status": "connected"}
        except Exception as e:
             return {"message": "pong", "redis_status": f"disconnected ({e})"}
    else:
        return {"message": "pong", "redis_status": "disabled"}

# --- Config Management Endpoints (Using Redis) ---

CONFIG_KEY_PREFIX = "bn_config:" # Use a prefix for keys in Redis

@app.post("/api/configs")
async def save_configuration(payload: SaveConfigPayload):
    """ Saves a graph configuration to Redis (Upstash) """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not available. Cannot save configuration.")

    config_id = f"{CONFIG_KEY_PREFIX}{uuid4()}" # Generate unique ID with prefix
    config_data = {
        "id": config_id, # Store ID within the data as well
        "name": payload.config_name,
        "graph_structure": payload.graph_structure.dict()
    }
    try:
        # Use redis_client.set to store the JSON string
        success = redis_client.set(config_id, json.dumps(config_data))
        if not success:
             raise Exception("Redis SET command failed to return success.")
        logger.info(f"Saved configuration '{payload.config_name}' with ID: {config_id}")
        return {"message": "Configuration saved successfully", "config_id": config_id, "config_name": payload.config_name}
    except Exception as e:
        logger.error(f"Error saving configuration to Redis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {e}")

@app.get("/api/configs")
async def list_configurations():
    """ Lists saved graph configurations from Redis (Upstash) """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not available. Cannot list configurations.")

    configs_summary = []
    try:
        # Use scan_iter to find keys matching the prefix without blocking
        # Note: scan_iter in redis-py is a generator
        config_keys = [key for key in redis_client.scan_iter(match=f"{CONFIG_KEY_PREFIX}*")]

        if not config_keys:
            logger.info("No configurations found in Redis.")
            return []

        # Fetch data for found keys. Use pipeline for potential efficiency if many keys.
        # For simplicity here, fetch one by one. Pipeline is better for >100 keys.
        for key in config_keys:
            try:
                config_json = redis_client.get(key)
                if config_json:
                    # Config data is already a string because decode_responses=True
                    config_data = json.loads(config_json)
                    configs_summary.append({
                        "id": config_data.get("id", key), # Prefer ID from data
                        "name": config_data.get("name", "Unnamed Config")
                    })
                else:
                     logger.warning(f"Key {key} found by SCAN but GET returned None.")
            except Exception as e:
                logger.warning(f"Failed to parse config data for key {key}: {e}")
                configs_summary.append({"id": key, "name": "Error loading name"})

        logger.info(f"Found {len(configs_summary)} configurations.")
        return configs_summary
    except Exception as e:
        logger.error(f"Error listing configurations from Redis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list configurations: {e}")

@app.get("/api/configs/{config_id}")
async def load_configuration(config_id: str):
    """ Loads a specific graph configuration from Redis (Upstash) """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not available. Cannot load configuration.")

     # Ensure the key has the prefix if sent without it
    if not config_id.startswith(CONFIG_KEY_PREFIX):
        logger.warning(f"Config ID '{config_id}' missing prefix, adding it.")
        config_id_with_prefix = f"{CONFIG_KEY_PREFIX}{config_id}"
    else:
        config_id_with_prefix = config_id


    try:
        config_json = redis_client.get(config_id_with_prefix)
        if config_json is None:
            raise HTTPException(status_code=404, detail=f"Configuration ID '{config_id_with_prefix}' not found.")

        # Already decoded to string
        config_data = json.loads(config_json)
        logger.info(f"Loaded configuration with ID: {config_id_with_prefix}")
        return config_data
    except HTTPException as e:
        raise e # Re-raise 404 etc.
    except json.JSONDecodeError as e:
         logger.error(f"Invalid JSON found for config ID '{config_id_with_prefix}': {e}")
         raise HTTPException(status_code=500, detail=f"Configuration data for '{config_id_with_prefix}' is corrupted.")
    except Exception as e:
        logger.error(f"Error loading configuration '{config_id_with_prefix}' from Redis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load configuration: {e}")


# --- Prediction Endpoint (Unchanged Logic, uses dynamic helpers) ---

@app.post("/api/predict_openai_bn_single_call")
async def predict_openai_bn_single_call(payload: PredictionPayload):
    """ Receives dynamic graph structure and input probabilities, returns all node probabilities estimated by the LLM. """
    logger.info("Entered /api/predict_openai_bn_single_call (dynamic) function.")
    try:
        input_probs_dict = payload.input_values
        graph_structure = payload.graph_structure
        node_parents, node_descriptions, node_types, all_nodes, target_nodes, input_nodes, relationship_descriptions = get_dynamic_node_info(graph_structure)
        estimated_target_probs = call_openai_dynamic_bn(input_probs_dict, graph_structure)
        all_current_probabilities = {**input_probs_dict, **estimated_target_probs}

        final_result = {}
        for node in all_nodes:
            node_desc = node_descriptions.get(node, node)
            if node in all_current_probabilities:
                p1 = all_current_probabilities[node]
                p1_clamped = max(0.0, min(1.0, p1))
                final_result[node] = {"0": 1.0 - p1_clamped, "1": p1_clamped, "description": node_desc}
            else:
                logger.warning(f"Node {node} not found in final probability dictionary - using default.")
                final_result[node] = {"0": 0.5, "1": 0.5, "description": node_desc}

        logger.info("Successfully generated prediction using dynamic graph.")
        # -- (Context preparation logic unchanged) --
        input_descriptions_list = []
        for node, value in input_probs_dict.items():
             if node in input_nodes:
                 state_desc = "High" if value >= 0.66 else ("Medium" if value >= 0.33 else "Low")
                 input_descriptions_list.append({ "node": node, "description": node_descriptions.get(node, node), "value": value, "state": state_desc })
        qualitative_rules_list = []
        # Simple pairing - might be inaccurate if relationship_descriptions structure changes
        for i in range(0, len(relationship_descriptions), 2):
             if i//2 < len(target_nodes): # Basic check
                 node_id = target_nodes[i//2]
                 desc = relationship_descriptions[i]
                 qual = relationship_descriptions[i+1].strip() if (i+1) < len(relationship_descriptions) else ""
                 qualitative_rules_list.append({"node": node_id, "description": desc, "qualitative": qual})

        return { "probabilities": final_result, "llm_context": { "input_states": input_descriptions_list, "node_dependencies": node_parents, "qualitative_rules": qualitative_rules_list, "node_descriptions": node_descriptions } }
    except HTTPException as e: raise e
    except Exception as e: logger.error(f"Error in dynamic prediction endpoint: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Root Endpoint ---
@app.get("/")
def root():
    logger.info("Entered / route (API function).")
    return {"message": "BN API is running. Frontend served from /public."}

print("DEBUG: Finished defining routes.", flush=True)
