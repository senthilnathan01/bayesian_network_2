// public/script.js

// --- Global Variables ---
let cy; // Cytoscape instance
let currentConfig = null; // Represents the currently loaded/active config object { id, name, graph_structure }
let defaultGraphStructure = null; // To store the structure fetched from backend
let sessionLog = []; // Logs for the current browser session
let edgeHandlesInstance = null; // Reference to edge handles extension API
let contextMenuInstance = null; // Reference to context menus extension API
let newNodeCounter = 0; // Counter for generating unique default node IDs
// Flags to track successful registration
let didRegisterEdgehandles = false;
let didRegisterContextMenus = false;

// --- DEFINE CORE HELPER FUNCTIONS FIRST ---

// Function to calculate node labels including probability
function nodeLabelFunc(node) {
    const id = node.data('id');
    const fullName = node.data('fullName') || id;
    const currentLabel = node.data('currentProbLabel'); // Set by updateNodeProbabilities
    return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`;
}

// Client-side cycle check helper (DFS based)
function simulateIsDag(graph) {
    if (!graph || !graph.nodes || !graph.edges) {
        console.warn("simulateIsDag: Invalid graph structure provided.");
        return false; // Cannot determine if it's a DAG
    }
    const adj = {};
    const nodesSet = new Set();
    // Initialize adjacency list and node set
    graph.nodes.forEach(n => {
        if (n && n.id) { // Basic check for valid node object
            adj[n.id] = [];
            nodesSet.add(n.id);
        } else {
             console.warn("simulateIsDag: Skipping invalid node object during init:", n);
        }
    });
    // Populate adjacency list from edges, ensuring source and target exist
    graph.edges.forEach(e => {
         if (e && e.source && e.target && nodesSet.has(e.source) && nodesSet.has(e.target)) {
            if (e.source in adj) {
                adj[e.source].push(e.target);
            } else {
                console.warn(`simulateIsDag: Source node ${e.source} not found in adj list during edge processing.`);
                 // If needed, initialize: adj[e.source] = [e.target];
                 // But ideally, all nodes should be in adj from the first loop.
            }
        } else {
            console.warn(`simulateIsDag: Skipping edge with missing/invalid node or edge object:`, e);
        }
    });

    const path = new Set();    // Nodes currently in the recursion path
    const visited = new Set(); // All nodes visited so far in DFS

    function dfs(node) {
        path.add(node);
        visited.add(node);

        for (const neighbor of adj[node] || []) {
            if (!nodesSet.has(neighbor)) { // Should be caught by edge check, but safe
                 console.warn(`simulateIsDag DFS: Neighbor ${neighbor} not in nodesSet.`);
                 continue;
             }
            if (path.has(neighbor)) { // Found a node already in the current path -> Cycle!
                console.log(`Cycle detected: ${neighbor} is already in path ${Array.from(path).join('->')}`);
                return false;
            }
            if (!visited.has(neighbor)) {
                if (!dfs(neighbor)) { // Recurse, if downstream finds a cycle, return false
                    return false;
                }
            }
        }
        path.delete(node); // Backtrack: remove node from current path
        return true;       // No cycle found starting from this node in this path
    }

    // Check all nodes to handle disconnected graphs
    for (const node of nodesSet) {
        if (!visited.has(node)) {
            if (!dfs(node)) {
                return false; // Cycle found
            }
        }
    }
    return true; // No cycles found in any component
}

// --- Initialization Sequence ---
document.addEventListener('DOMContentLoaded', async () => {
    console.log("DOM Content Loaded. Starting initialization...");

    // Check core library and container first
    if (typeof cytoscape !== 'function') { alert("Error: Cytoscape library failed to load."); setStatusMessage('config-status', "Error: Core graph library failed.", "error"); showSpinner(false); return; }
    if (!document.getElementById('cy')) { alert("Error: Graph container element 'cy' not found."); setStatusMessage('config-status', "Error: Graph container missing.", "error"); showSpinner(false); return; }

    // 1. Initialize Cytoscape Instance
    initializeCytoscape(); // Creates 'cy' instance

    if (!cy) { showSpinner(false); return; } // Stop if core failed (alert shown inside init)

    // 2. Initialize Editing Extensions (includes registration check)
    initializeEditingExtensions(); // Tries to register and init extensions on 'cy'

    // 3. Initialize UI Button Listeners
    initializeUI();

    // 4. Load Data (Default & Saved)
    showSpinner(true);
    setStatusMessage('config-status', "Loading config data...", "loading");
    try {
        await loadDefaultConfig(); // Fetch default structure
        if (defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true); // Load default into cy
            addDefaultToDropdown();
            const select = document.getElementById('load-config-select');
            if(!select.value || select.value === "") { selectConfigInDropdown(defaultGraphStructure.id); }
        } else { throw new Error("Default config data unavailable."); }

        await loadConfigList(); // Fetch saved configs
        const finalStatus = currentConfig ? `Config '${currentConfig.name}' loaded.` : "Ready.";
        // Set status message correctly based on whether default loaded
        setStatusMessage('config-status', finalStatus, "success");

    } catch (error) {
        console.error("Initialization Data Load Error:", error);
        setStatusMessage('config-status', `Initialization failed: ${error.message}`, "error");
        if (!currentConfig) { updateInputControls([]); document.getElementById('current-config-name').textContent = "None"; }
    } finally {
        showSpinner(false);
    }
});

// --- Cytoscape Core Initialization ---
function initializeCytoscape() {
    console.log("Attempting Cytoscape core initialization...");
    try {
        // Ensure nodeLabelFunc is defined before this point
        if (typeof nodeLabelFunc !== 'function') {
             throw new Error("Internal Error: nodeLabelFunc is not defined before Cytoscape init.");
        }
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: [],
            style: [ // Ensure nodeLabelFunc is accessible here
                { selector: 'node', style: { 'background-color': '#ccc', 'label': nodeLabelFunc, 'width': 120, 'height': 120, 'shape': 'ellipse', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px', 'font-weight': '100', 'text-wrap': 'wrap', 'text-max-width': 110, 'text-outline-color': '#fff', 'text-outline-width': 1, 'color': '#333', 'transition-property': 'background-color, color', 'transition-duration': '0.5s' } },
                { selector: 'node[nodeType="input"]', style: { 'shape': 'rectangle', 'width': 130, 'height': 70 } },
                { selector: 'node[nodeType="hidden"]', style: { 'shape': 'ellipse' } },
                { selector: 'edge', style: { 'width': 2, 'line-color': '#666', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#666', 'curve-style': 'bezier' } },
                { selector: '.edge-source-active', style: { 'border-width': 3, 'border-color': '#00ff00' } },
                { selector: '.eh-grabbed', style: { 'border-width': 3, 'border-color': '#007bff' } }
            ],
            layout: { name: 'preset' } // Use preset initially, run layout later
        });
        console.log("Cytoscape core initialized successfully.");
    } catch(error) {
        console.error("Failed to initialize Cytoscape Core:", error);
        // Show detailed error to user
        alert(`Critical Error initializing graph library: ${error.message}\n\n(Check console for more details)`);
        cy = null; // Ensure cy is null if init fails
    }
}

// --- Editing Extensions Initialization & REGISTRATION ---
function initializeEditingExtensions() {
    console.log("Initializing and Registering editing extensions...");
    if (!cy) { console.error("Cannot init extensions: Cytoscape not ready."); return; }

    // --- Register & Initialize Edge Handles ---
    try {
        // Check if library function exists globally AND registration hasn't happened
        if (typeof cytoscapeEdgehandles === 'function' && !didRegisterEdgehandles) {
             cytoscape.use( cytoscapeEdgehandles ); // REGISTER the extension
             didRegisterEdgehandles = true; // Mark as registered only once
             console.log("cytoscape-edgehandles registered.");

            // Now initialize on the instance
            edgeHandlesInstance = cy.edgehandles({ // INITIALIZE instance
                 snap: true, handleNodes: 'node', handleSize: 10, handleColor: '#007bff', handlePosition: 'middle top', preview: true, hoverDelay: 150,
                 edgeType: (src, tgt) => 'flat', loopAllowed: (n) => false, nodeLoopOffset: -50,
                 nodeParams: (src, tgt) => ({}), edgeParams: (src, tgt, i) => ({ data: { source: src.id(), target: tgt.id() } }), ghostEdgeParams: () => ({}),
                 handleClass: 'eh-handle', hoverClass: 'eh-hover', snapClass: 'eh-snap', sourceNodeClass: 'eh-source', targetNodeClass: 'eh-target', ghostEdgeClass: 'eh-ghost-edge', previewClass: 'eh-preview', grabbedNodeClass: 'eh-grabbed',
                 complete: ( src, tgt, added ) => {
                     console.log("Edge draw completed between:", src.id(), "and", tgt.id());
                     // The edge (added) is already added to the graph by the extension by default
                     // We need to validate it immediately
                     const currentEdges = cy.edges().map(e=>({source:e.source().id(), target:e.target().id()}));
                     const currentNodes = cy.nodes().map(n=>n.data()); // Need node data too
                     const currentGraph = { nodes: currentNodes, edges: currentEdges };

                     if (!simulateIsDag(currentGraph)) {
                          alert(`Cycle detected! Removing edge ${src.id()}->${tgt.id()}`);
                          cy.remove(added); // Remove the edge that caused the cycle
                          setStatusMessage('config-status',`Edge aborted (cycle).`,"error");
                      } else {
                          markConfigUnsaved();
                          setStatusMessage('config-status',`Edge ${src.id()}->${tgt.id()} added.`,"success");
                          // Maybe run layout? Optional.
                          // runLayout();
                      }
                  }
             });
            console.log("Edgehandles instance initialized.");
        } else if (!didRegisterEdgehandles) {
             console.warn("cytoscape-edgehandles lib function not found globally. Cannot register or initialize.");
        } else {
            console.log("Edgehandles already registered and likely initialized.");
        }
    } catch (e) {
        console.error("Error during Edgehandles registration/initialization:", e);
        edgeHandlesInstance = null; // Ensure instance is null on error
    }

    // --- Register & Initialize Context Menus ---
    try {
        // Check dependencies AND library function AND registration status
        if (typeof cytoscapeContextMenus === 'function' && typeof tippy === 'function' && typeof Popper === 'object' && !didRegisterContextMenus) {
             cytoscape.use( cytoscapeContextMenus ); // REGISTER the extension
             didRegisterContextMenus = true; // Mark as registered only once
             console.log("cytoscape-context-menus registered.");

            // Now initialize on the instance
             contextMenuInstance = cy.contextMenus({ // INITIALIZE instance
                 menuItems: [ // Use 'content' for V4.x
                     { id: 'add-input-core', content: 'Add Input Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('input', evt.position || evt.cyPosition) },
                     { id: 'add-hidden-core', content: 'Add Hidden Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('hidden', evt.position || evt.cyPosition), hasTrailingDivider: true },
                     { id: 'convert-type', content: 'Convert Type', selector: 'node', onClickFunction: (evt) => convertNodeType(evt.target || evt.cyTarget) },
                     { id: 'delete-node', content: 'Delete Node', selector: 'node', hasTrailingDivider: true, onClickFunction: (evt) => deleteElement(evt.target || evt.cyTarget) },
                     { id: 'delete-edge', content: 'Delete Edge', selector: 'edge', onClickFunction: (evt) => deleteElement(evt.target || evt.cyTarget) }
                 ],
                 menuItemClasses: [ 'ctx-menu-item' ], contextMenuClasses: [ 'ctx-menu' ]
             });
            console.log("Context Menus instance initialized.");
        } else if (!didRegisterContextMenus) {
             console.warn("cytoscape-context-menus lib or dependencies (Tippy/Popper) not found globally. Cannot register or initialize.");
        } else {
            console.log("Context Menus already registered and likely initialized.");
        }
    } catch (e) {
        console.error("Error during Context Menus registration/initialization:", e);
        contextMenuInstance = null; // Ensure instance is null on error
    }

     // Update placeholder text based on actual initialization success
     updateEditorPlaceholderText();
}

// --- UI Initialization ---
function initializeUI() {
    console.log("Initializing UI event listeners...");
    document.getElementById('save-config-button')?.addEventListener('click', saveConfiguration);
    document.getElementById('load-config-button')?.addEventListener('click', loadSelectedConfiguration);
    document.getElementById('set-default-button')?.addEventListener('click', setDefaultConfiguration);
    document.getElementById('delete-config-button')?.addEventListener('click', deleteSelectedConfiguration);
    document.getElementById('update-button')?.addEventListener('click', fetchAndUpdateLLM);
    document.getElementById('gradient-toggle')?.addEventListener('change', () => updateNodeProbabilities(null));
    document.getElementById('download-session-log-button')?.addEventListener('click', downloadSessionLog);
    document.getElementById('download-all-log-button')?.addEventListener('click', downloadAllLogs);
    console.log("UI Listeners attached.");
}

// --- Editing Functions ---
function addNodeFromMenu(nodeType, position) { if (!cy) return; newNodeCounter++; const idBase = nodeType === 'input' ? 'Input' : 'Hidden'; let id = `${idBase}_${newNodeCounter}`; while (cy.getElementById(id).length > 0) { newNodeCounter++; id = `${idBase}_${newNodeCounter}`; } const name = prompt(`Enter display name for new ${nodeType} node (ID: ${id}):`, id); if (name === null) return; const newNodeData = { group: 'nodes', data: { id: id.trim(), fullName: name.trim() || id.trim(), nodeType: nodeType }, position: position }; cy.add(newNodeData); console.log(`Added ${nodeType} node: ID=${id}, Name=${newNodeData.data.fullName}`); if (nodeType === 'input') { updateInputControls(cy.nodes().map(n => n.data())); } updateNodeProbabilities({}); runLayout(); markConfigUnsaved(); }
function deleteElement(target) { if (!cy || !target || (!target.isNode && !target.isEdge)) return; const id = target.id(); const type = target.isNode() ? 'node' : 'edge'; let name = target.data('fullName') || id; if (target.isEdge()){ name = `${target.source().id()}->${target.target().id()}`; } if (confirm(`Delete ${type} "${name}"?`)) { const wasInputNode = target.isNode() && target.data('nodeType') === 'input'; cy.remove(target); console.log(`Removed ${type}: ${id}`); if(wasInputNode){ updateInputControls(cy.nodes().map(n => n.data())); } markConfigUnsaved(); } }
function convertNodeType(targetNode) { if (!cy || !targetNode || !targetNode.isNode()) return; const currentType = targetNode.data('nodeType'); const newType = currentType === 'input' ? 'hidden' : 'input'; const nodeId = targetNode.id(); if (confirm(`Convert node "${targetNode.data('fullName') || nodeId}" from ${currentType} to ${newType}?`)) { targetNode.data('nodeType', newType); console.log(`Converted node ${nodeId} to ${newType}`); cy.style().update(); updateInputControls(cy.nodes().map(n => n.data())); updateNodeProbabilities({}); markConfigUnsaved(); } }
function markConfigUnsaved() { const d = document.getElementById('current-config-name'); if (d && !d.textContent.endsWith('*')) { d.textContent += '*'; setStatusMessage('config-status', "Graph modified. Save changes.", "loading"); } }
function clearUnsavedMark() { const d = document.getElementById('current-config-name'); if (d && d.textContent.endsWith('*')) { d.textContent = d.textContent.slice(0, -1); } }


// --- Configuration Management Functions ---
async function loadDefaultConfig() { console.log("Fetching default configuration structure..."); try { await retryFetch(async () => { const r = await fetch('/api/configs/default'); if (!r.ok) throw new Error(`HTTP ${r.status}`); defaultGraphStructure = await r.json(); console.log("Default structure stored."); }, 3); } catch (e) { console.error('Error fetching default struct:', e); defaultGraphStructure = null; throw e; } }
function addDefaultToDropdown() { const s=document.getElementById('load-config-select'); if(!defaultGraphStructure)return; let e=false; for(let i=0;i<s.options.length;i++){if(s.options[i].value===defaultGraphStructure.id){e=true;break;}} if(!e){const o=document.createElement('option');o.value=defaultGraphStructure.id;o.textContent=`${defaultGraphStructure.name} (Default)`; if(s.options.length>0&&s.options[0].value===""){s.insertBefore(o,s.options[1]);}else{s.insertBefore(o,s.firstChild);} console.log("Added Default to dropdown.");} }
function loadGraphData(configData, isDefault=false) { if(!cy||!configData||!configData.graph_structure){console.error("Load graph error"); return;} console.log(`Loading: ${configData.name}`); const elems = configData.graph_structure.nodes.map(n=>({data:{id:n.id,fullName:n.fullName,nodeType:n.nodeType}})).concat(configData.graph_structure.edges.map(e=>({data:{source:e.source,target:e.target}}))); cy.elements().remove(); cy.add(elems); runLayout(); currentConfig={...configData}; document.getElementById('config-name').value=isDefault?'':currentConfig.name; document.getElementById('current-config-name').textContent=currentConfig.name; updateInputControls(currentConfig.graph_structure.nodes); updateNodeProbabilities({}); clearSessionLog(); clearLLMOutputs(); clearUnsavedMark(); selectConfigInDropdown(currentConfig.id); }

async function saveConfiguration() { const ni=document.getElementById('config-name'); const n=ni.value.trim(); if(!n){setStatusMessage('config-status',"Enter name.","error");return;} if(!cy){setStatusMessage('config-status',"Graph not init.","error");return;} setStatusMessage('config-status',"Saving...","loading"); showSpinner(true); enableUI(false); const struct = {nodes: cy.nodes().map(node=>({id:node.id(),fullName:node.data('fullName'),nodeType:node.data('nodeType')})), edges: cy.edges().map(edge=>({source:edge.source().id(),target:edge.target().id()}))}; await retryFetch(async()=>{ const r=await fetch('/api/configs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config_name:n,graph_structure:struct})}); const d=await r.json(); if(!r.ok)throw new Error(d.detail||`HTTP ${r.status}`); currentConfig={id:d.config_id,name:d.config_name,graph_structure:struct}; document.getElementById('current-config-name').textContent=currentConfig.name; ni.value=''; setStatusMessage('config-status',`Saved as '${d.config_name}'.`,"success"); await loadConfigList(); selectConfigInDropdown(d.config_id); clearSessionLog(); clearUnsavedMark();},3,()=>setStatusMessage('config-status',"Save fail. Retry...","error")).catch(e=>{console.error('Save error:',e); setStatusMessage('config-status',`Save fail: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);}); }
async function loadConfigList() { console.log("Updating saved list..."); try { await retryFetch(async()=>{ const r=await fetch('/api/configs'); if(!r.ok)throw new Error(`HTTP ${r.status}`); const configs=await r.json(); const s=document.getElementById('load-config-select'); const curSel=s.value; const optsToRemove=[]; for(let i=0;i<s.options.length;i++){const v=s.options[i].value; if(v!==""&&v!=="default-config-001"){optsToRemove.push(s.options[i]);}} optsToRemove.forEach(opt=>s.removeChild(opt)); configs.forEach(c=>{const o=document.createElement('option');o.value=c.id;o.textContent=c.name;s.appendChild(o);}); s.value=curSel; let foundCur=false; if(currentConfig){for(let i=0;i<s.options.length;i++){if(s.options[i].value===currentConfig.id){s.value=currentConfig.id;foundCur=true;break;}}} if(!foundCur&&defaultGraphStructure){s.value=defaultGraphStructure.id;} if(s.value==="")s.selectedIndex=0; console.log("Saved list updated.");},3); } catch(e){ console.error('Load list error:',e); setStatusMessage('config-status',`Failed list load: ${e.message}`,"error");}}
async function loadSelectedConfiguration() { const s=document.getElementById('load-config-select'); const id=s.value; if(!id){setStatusMessage('config-status',"Select config.","error");return;} if(!cy){setStatusMessage('config-status',"Graph not ready.","error");return;} setStatusMessage('config-status',`Loading ${s.options[s.selectedIndex].text}...`,"loading"); showSpinner(true); enableUI(false); await retryFetch(async()=>{ if(id==='default-config-001'&&defaultGraphStructure){loadGraphData(defaultGraphStructure,true);setStatusMessage('config-status',"Default loaded.","success");return;} const r=await fetch(`/api/configs/${id}`); if(!r.ok){const e=await r.json().catch(()=>({detail:`HTTP ${r.status}`}));throw new Error(e.detail||`HTTP ${r.status}`);} const d=await r.json(); loadGraphData(d,false); setStatusMessage('config-status',`Config '${d.name}' loaded.`,"success");},3,()=>setStatusMessage('config-status',"Load fail. Retry...","error")).catch(e=>{console.error('Load select error:',e); setStatusMessage('config-status',`Load failed: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);});}
async function setDefaultConfiguration() { const s=document.getElementById('load-config-select'); const id=s.value; const name=s.options[s.selectedIndex].text; if(!id||id==='default-config-001'){setStatusMessage('config-status',"Select saved config.","error");return;} if(!confirm(`Set "${name}" as default?`))return; setStatusMessage('config-status',`Setting default...`,"loading"); showSpinner(true); enableUI(false); await retryFetch(async()=>{ const r=await fetch('/api/configs/set_default',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(id)}); const d=await r.json(); if(!r.ok)throw new Error(d.detail||`HTTP ${r.status}`); setStatusMessage('config-status',`Set '${name}' as default.`,"success"); await loadDefaultConfig();},3,()=>setStatusMessage('config-status',"Set default fail. Retry...","error")).catch(e=>{console.error('Set default error:',e);setStatusMessage('config-status',`Set default failed: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);});}
async function deleteSelectedConfiguration() { const s=document.getElementById('load-config-select'); const id=s.value; const name=s.options[s.selectedIndex].text; if(!id||id==='default-config-001'){setStatusMessage('config-status',"Select saved config.","error");return;} if(!confirm(`Delete "${name}"?`))return; setStatusMessage('config-status',`Deleting...`,"loading"); showSpinner(true); enableUI(false); await retryFetch(async()=>{ const r=await fetch(`/api/configs/${id}`,{method:'DELETE'}); const d=await r.json(); if(!r.ok)throw new Error(d.detail||`HTTP ${r.status}`); setStatusMessage('config-status',`Deleted '${name}'.`,"success"); if(currentConfig && currentConfig.id===id){if(defaultGraphStructure){loadGraphData(defaultGraphStructure,true);setStatusMessage('config-status',`Deleted '${name}'. Default loaded.`,"success");}else{cy.elements().remove();currentConfig=null;updateInputControls([]);clearSessionLog();clearLLMOutputs();document.getElementById('current-config-name').textContent="None";setStatusMessage('config-status',`Deleted '${name}'. Load another.`,"success");}} await loadConfigList();},3,()=>setStatusMessage('config-status',"Delete fail. Retry...","error")).catch(e=>{console.error('Delete error:',e); setStatusMessage('config-status',`Delete failed: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);});}

// --- Other Utility Functions ---
function updateEditorPlaceholderText() { const p=document.querySelector('.graph-editor p'); if(!p)return; let m=""; const menusOk=contextMenuInstance!==null; const edgesOk=edgeHandlesInstance!==null; if(!menusOk&&!edgesOk) m="Editing unavailable: Failed libraries."; else if(!menusOk) m="Right-click menus disabled. Drag handles."; else if(!edgesOk) m="Edge drawing disabled. Use right-click."; else m="Right-click to edit. Drag handles for edges."; p.textContent=m; p.style.color=(menusOk&&edgesOk)?'#555':'orange'; }

// ... (fetchAndUpdateLLM, updateNodeProbabilities, LLM display, logging, downloads - paste these full functions here) ...
// --- Prediction ---
async function fetchAndUpdateLLM() { if (!currentConfig || !currentConfig.graph_structure || !currentConfig.graph_structure.nodes.length === 0) { alert("Load config first."); return; } if (!cy) { alert("Graph not ready."); return; } setStatusMessage('predict-status', "Gathering inputs...", "loading"); let inputs = { input_values: {} }; let invalid = false; currentConfig.graph_structure.nodes.filter(n => n.nodeType === 'input').forEach(n => { const el = document.getElementById(`input-${n.id}`); const cont = el?.parentElement; let v = 0.5; if (el) { v = parseFloat(el.value); if (isNaN(v) || v < 0 || v > 1) { setStatusMessage('predict-status', `Invalid input for ${n.id}.`, "error"); cont?.classList.add('invalid-input'); invalid = true; } else { cont?.classList.remove('invalid-input'); } } inputs.input_values[n.id] = isNaN(v)?0.5:Math.max(0, Math.min(1, v)); }); if (invalid) { setStatusMessage('predict-status', "Fix invalid inputs (0-1).", "error"); return; } const payload = { ...inputs, graph_structure: currentConfig.graph_structure, config_id: currentConfig.id, config_name: currentConfig.name }; setStatusMessage('predict-status', "Running prediction...", "loading"); showSpinner(true); enableUI(false); clearLLMOutputs(); await retryFetch(async () => { const r = await fetch('/api/predict_openai_bn_single_call', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); const d = await r.json(); if (!r.ok) throw new Error(d.detail || `HTTP error ${r.status}`); updateNodeProbabilities(d.probabilities); displayLLMReasoning(d.llm_reasoning); displayLLMContext(d.llm_context); setStatusMessage('predict-status', "Prediction complete.", "success"); logPrediction(inputs.input_values, d.probabilities); }, 3, () => setStatusMessage('predict-status', "Prediction failed. Retrying...", "error")).catch(e => { console.error("Predict error:", e); setStatusMessage('predict-status', `Prediction failed: ${e.message}`, "error"); clearLLMOutputs(); }).finally(() => { enableUI(true); showSpinner(false); }); }
// --- Node Appearance Update ---
function updateNodeProbabilities(probabilities) { if (!cy) return; const useGradient = document.getElementById('gradient-toggle').checked; cy.nodes().forEach(node => { const nodeId = node.id(); const nodeType = node.data('nodeType'); let probState1 = null; if (probabilities && probabilities[nodeId] && probabilities[nodeId]["1"] !== undefined) { probState1 = probabilities[nodeId]["1"]; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); } else if (nodeType === 'input') { const inputElement = document.getElementById(`input-${nodeId}`); const currentVal = inputElement ? (parseFloat(inputElement.value) || 0.5) : 0.5; probState1 = currentVal; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); } else { node.data('currentProbLabel', '(N/A)'); } let baseBgColor = nodeType === 'input' ? '#add8e6' : '#f0e68c'; let textColor = '#333'; let finalBgColor = baseBgColor; if (probState1 !== null) { if (useGradient) { finalBgColor = `rgb(${Math.round(255 * (1 - probState1))}, ${Math.round(255 * probState1)}, 0)`; textColor = '#333'; } else { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; } } else if (!useGradient && nodeType !== 'input') { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; } node.style({ 'background-color': finalBgColor, 'color': textColor }); }); cy.style().update(); }
// --- LLM Output Display ---
function displayLLMReasoning(reasoningText) { const d=document.getElementById('llm-reasoning-content'); d.textContent=reasoningText||"N/A";}
function displayLLMContext(context) { if(!context)return; const iD=document.getElementById('input-context'); const sD=document.getElementById('structure-context'); const dD=document.getElementById('node-descriptions-context'); let iH='<ul>';(context.input_states||[]).forEach(s=>{iH+=`<li>${s.node}(${s.description}): ${s.state} (p=${s.value.toFixed(2)})</li>`;});iH+='</ul>';iD.innerHTML=iH||'N/A'; let sH='<ul>';Object.entries(context.node_dependencies||{}).forEach(([n,p])=>{sH+=`<li>${n}: ${p.join(',')||'None'}</li>`;});sH+='</ul>';sD.innerHTML=sH||'N/A'; let dH='<ul>';Object.entries(context.node_descriptions||{}).forEach(([n,d])=>{dH+=`<li>${n}: ${d}</li>`;});dH+='</ul>';dD.innerHTML=dH||'N/A';}
function clearLLMOutputs() { document.getElementById('llm-reasoning-content').textContent = 'Run prediction...'; document.getElementById('input-context').innerHTML = '<p>N/A</p>'; document.getElementById('structure-context').innerHTML = '<p>N/A</p>'; document.getElementById('node-descriptions-context').innerHTML = '<p>N/A</p>'; setStatusMessage('predict-status', "", ""); }
// --- Logging ---
function logPrediction(inputs, probabilities) { const logEntry = { timestamp: new Date().toISOString(), configId: currentConfig ? currentConfig.id : "unknown", configName: currentConfig ? currentConfig.name : "Unknown", inputs: { ...inputs }, probabilities: {} }; for (const n in probabilities) { logEntry.probabilities[n] = probabilities[n]["1"]; } for (const i in inputs) { if (!(i in logEntry.probabilities)) { logEntry.probabilities[i] = inputs[i]; } } sessionLog.push(logEntry); document.getElementById('log-count').textContent = `Session logs: ${sessionLog.length}`; }
function clearSessionLog() { sessionLog = []; document.getElementById('log-count').textContent = `Session logs: 0`; }
function downloadSessionLog() { if(sessionLog.length===0){alert("No logs.");return;} const h=['Timestamp','ConfigID','ConfigName','NodeID','ProbP1']; const r=[]; sessionLog.forEach(l=>{ const probs = l.probabilities || {}; Object.entries(probs).forEach(([n,p])=>{ if(typeof p === 'number') r.push([l.timestamp,l.configId,l.configName,n,p.toFixed(4)]); else console.warn("Skip non-numeric prob:", n); }); }); const csv=Papa.unparse({fields:h,data:r}); triggerCsvDownload(csv, `session_log_${(currentConfig?.name||'unsaved').replace(/[^a-z0-9]/gi,'_')}`); }
async function downloadAllLogs() { if(!currentConfig||!currentConfig.id||currentConfig.id==="unknown"||currentConfig.id==="default-config-001"){alert("Load saved config.");return;} setStatusMessage('predict-status',"Downloading logs...","loading"); showSpinner(true); enableUI(false); await retryFetch(async()=>{ const r = await fetch(`/api/download_log/${currentConfig.id}`); if(!r.ok){ if(r.status === 404) throw new Error(`No logs for '${currentConfig.name}'.`); const e = await r.json().catch(()=>({detail:`HTTP ${r.status}`})); throw new Error(e.detail); } const b = await r.blob(); triggerCsvDownload(b, `all_logs_${currentConfig.name.replace(/[^a-z0-9]/gi,'_')}`); setStatusMessage('predict-status',"Logs downloaded.","success"); },3,()=>setStatusMessage('predict-status',"Download fail. Retry...","error")).catch(e=>{setStatusMessage('predict-status',`Log download fail: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);});}
function triggerCsvDownload(csvDataOrBlob, baseFilename) { const blob = (csvDataOrBlob instanceof Blob) ? csvDataOrBlob : new Blob([csvDataOrBlob], { type: 'text/csv;charset=utf-8;' }); const link = document.createElement("a"); const url = URL.createObjectURL(blob); link.setAttribute("href", url); const timestampStr = new Date().toISOString().replace(/[:.]/g, '-'); link.setAttribute("download", `${baseFilename}_${timestampStr}.csv`); link.style.visibility = 'hidden'; document.body.appendChild(link); link.click(); document.body.removeChild(link); URL.revokeObjectURL(url); }


console.log("script.js loaded and execution potentially started (check DOMContentLoaded).");
