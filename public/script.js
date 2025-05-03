// public/script.js

console.log("script.js: Parsing script...");

// --- Global Variables ---
let cy;
let currentConfig = null;
let defaultGraphStructure = null;
let sessionLog = [];
let contextMenuInstance = null;
let newNodeCounter = 0;
let didRegisterContextMenus = false;
let edgeSourceNode = null;
// NEW State Variables
let currentUiFeatures = null; // Holds results from /api/extract_ui_features
let currentUserPerception = ""; // Holds text from stage 2

// ===========================================
// --- ALL HELPER FUNCTION DEFINITIONS ---
// ===========================================
function nodeLabelFunc(node) {
    const id = node.data('id');
    const fullName = node.data('fullName') || id;
    const currentLabel = node.data('currentProbLabel');
    return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`;
}

function simulateIsDag(graph) {
    if (!graph || !graph.nodes || !graph.edges) { console.warn("simulateIsDag: Invalid graph structure."); return false; }
    const adj = {}; const nodesSet = new Set();
    graph.nodes.forEach(n => { if (n && n.id) { adj[n.id] = []; nodesSet.add(n.id); } else { console.warn("simulateIsDag: Invalid node object:", n); } });
    graph.edges.forEach(e => { if (e && e.source && e.target && nodesSet.has(e.source) && nodesSet.has(e.target)) { if (e.source in adj) { adj[e.source].push(e.target); } else { adj[e.source] = [e.target]; console.warn(`simulateIsDag: Source ${e.source} init`); } } else { console.warn(`simulateIsDag: Invalid edge:`, e); } });
    const path = new Set(); const visited = new Set();
    function dfs(node) { path.add(node); visited.add(node); for (const neighbor of adj[node] || []) { if (!nodesSet.has(neighbor)) continue; if (path.has(neighbor)) { console.log(`Cycle: ${neighbor} in ${Array.from(path).join('->')}`); return false; } if (!visited.has(neighbor)) { if (!dfs(neighbor)) return false; } } path.delete(node); return true; }
    for (const node of nodesSet) { if (!visited.has(node)) { if (!dfs(node)) return false; } }
    return true;
}

function selectConfigInDropdown(configId) { const select = document.getElementById('load-config-select'); if (select) select.value = configId; }
function enableUI(enable) { const buttons = document.querySelectorAll('button'); const inputs = document.querySelectorAll('input, select'); buttons.forEach(btn => btn.disabled = !enable); inputs.forEach(inp => inp.disabled = !enable); const toggle = document.getElementById('gradient-toggle'); if(toggle) toggle.disabled = false; document.body.style.cursor = enable ? 'default' : 'wait'; }
function showSpinner(show) { const spinner = document.getElementById('loading-spinner'); if (spinner) spinner.classList.toggle('hidden', !show); }
function runLayout() { if (!cy) return; let l = 'cola'; try { cy.layout({ name: 'cola', animate:true, nodeSpacing: 50, edgeLength: 180, padding: 30 }).run(); } catch (e) { console.warn('Cola failed, using dagre'); l = 'dagre'; try { cy.layout({ name: 'dagre', rankDir:'TB', spacingFactor: 1.2 }).run(); } catch (e) { console.error('Layouts failed'); l = 'grid'; try { cy.layout({ name: 'grid' }).run(); } catch(e) { console.error("Grid layout failed too!")}}} console.log('Using layout:', l); }
// function updateInputControls(nodes) { /* REMOVED - Replaced by updatePersonaInputControls */ }
function updateNodeProbabilities(probabilities) { if (!cy) return; const useGradient = document.getElementById('gradient-toggle')?.checked ?? true; cy.nodes().forEach(node => { const nodeId = node.id(); const nodeType = node.data('nodeType'); let probState1 = null; if (probabilities && probabilities[nodeId] && probabilities[nodeId]["1"] !== undefined) { probState1 = probabilities[nodeId]["1"]; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); } else if (nodeType === 'input') { const inputElement = document.getElementById(`input-${nodeId}`); const currentVal = inputElement ? (parseFloat(inputElement.value) || 0.5) : 0.5; probState1 = currentVal; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); } else { node.data('currentProbLabel', '(N/A)'); } let baseBgColor = nodeType === 'input' ? '#add8e6' : '#f0e68c'; let textColor = '#333'; let finalBgColor = baseBgColor; if (probState1 !== null) { if (useGradient) { finalBgColor = `rgb(${Math.round(255 * (1 - probState1))}, ${Math.round(255 * probState1)}, 0)`; textColor = '#333'; } else { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; } } else if (!useGradient && nodeType !== 'input') { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; } node.style({ 'background-color': finalBgColor, 'color': textColor }); }); cy.style().update(); }
function displayLLMReasoning(reasoningText) { const d=document.getElementById('llm-reasoning-content'); if(d) d.textContent=reasoningText||"N/A";}
// function displayLLMContext(context) { /* REMOVED - Replaced by displayDebugContext */ }
function setStatusMessage(elementId, message, type) { const el=document.getElementById(elementId); if(el){el.textContent=message; el.className=`status-message ${type||''}`;}}
async function retryFetch(fetchFn, maxRetries, onRetry) { let lastError; for(let attempt=1; attempt<=maxRetries; attempt++){ try{ await fetchFn(); return; }catch(error){ lastError=error; console.warn(`Attempt ${attempt} fail: ${error.message}`); if(attempt<maxRetries){ if(onRetry)onRetry(); await new Promise(resolve=>setTimeout(resolve, 1000*attempt));}}} throw lastError; }
function clearSessionLog() { sessionLog = []; const logCountEl = document.getElementById('log-count'); if (logCountEl) logCountEl.textContent = `Session logs: 0`; }
function clearLLMOutputs() { const reasonEl=document.getElementById('llm-reasoning-content'); if(reasonEl) reasonEl.textContent='Run prediction...'; const ic=document.getElementById('input-context'); if(ic) ic.innerHTML='<p>N/A</p>'; const sc=document.getElementById('structure-context'); if(sc) sc.innerHTML='<p>N/A</p>'; const dc=document.getElementById('node-descriptions-context'); if(dc) dc.innerHTML='<p>N/A</p>'; setStatusMessage('predict-status', "", ""); }
// function downloadSessionLog() { /* MODIFIED Below */ }
async function downloadAllLogs() { if(!currentConfig||!currentConfig.id||currentConfig.id==="unknown"||currentConfig.id==="default-config-001"){alert("Load saved config.");return;} setStatusMessage('predict-status',"Downloading logs...","loading"); showSpinner(true); enableUI(false); await retryFetch(async()=>{ const r = await fetch(`/api/download_log/${currentConfig.id}`); if(!r.ok){ if(r.status === 404) throw new Error(`No logs for '${currentConfig.name}'.`); const e = await r.json().catch(()=>({detail:`HTTP ${r.status}`})); throw new Error(e.detail); } const b = await r.blob(); triggerCsvDownload(b, `all_logs_${currentConfig.name.replace(/[^a-z0-9]/gi,'_')}`); setStatusMessage('predict-status',"Logs downloaded.","success"); },3,()=>setStatusMessage('predict-status',"Download fail. Retry...","error")).catch(e=>{setStatusMessage('predict-status',`Log download fail: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);});}
function triggerCsvDownload(csvDataOrBlob, baseFilename) { const blob = (csvDataOrBlob instanceof Blob) ? csvDataOrBlob : new Blob([csvDataOrBlob], { type: 'text/csv;charset=utf-8;' }); const link = document.createElement("a"); const url = URL.createObjectURL(blob); link.setAttribute("href", url); const timestampStr = new Date().toISOString().replace(/[:.]/g, '-'); link.setAttribute("download", `${baseFilename}_${timestampStr}.csv`); link.style.visibility = 'hidden'; document.body.appendChild(link); link.click(); document.body.removeChild(link); URL.revokeObjectURL(url); }
function markConfigUnsaved() { const d = document.getElementById('current-config-name'); if (d && !d.textContent.endsWith('*')) { d.textContent += '*'; setStatusMessage('config-status', "Graph modified. Save changes.", "loading"); } }
function clearUnsavedMark() { const d = document.getElementById('current-config-name'); if (d && d.textContent.endsWith('*')) { d.textContent = d.textContent.slice(0, -1); } }
function updateEditorPlaceholderText() {
    const p = document.querySelector('.graph-editor p');
    if (!p) {
        console.error("Could not find graph editor instruction paragraph element.");
        return;
    }

    // --- Detailed Instructions ---
    p.innerHTML = `
        <b>Editing Actions (Right-Click):</b><br>
           • <b>On Empty Canvas:</b> Choose 'Add Input Node' or 'Add Hidden Node'. You'll be prompted for a display name (ID is auto-generated).<br>
           • <b>On a Node:</b>
            <ul>
                <li><i>Start Edge From Here:</i> Select this, then <b>left-click</b> the target node to create a connection (cycles are prevented). Click canvas to cancel.</li>
                <li><i>Convert Type:</i> Toggles node between 'Input' (rectangle) and 'Hidden' (ellipse).</li>
                <li><i>Delete Node:</i> Removes the node and any connected edges.</li>
            </ul>
           • <b>On an Edge:</b> Choose 'Delete Edge' to remove the connection.
        <br>
        <i>Remember to 'Save' your configuration after making changes!</i>
    `;
    p.style.color = '#555'; // Reset color to default
    p.style.textAlign = 'left'; // Align text left for readability
    p.style.fontSize = '0.85em'; // Slightly smaller font for instructions

    // Optional: Still log warning if menus failed
    const menusOk = contextMenuInstance !== null;
    if (!menusOk) {
        console.warn("Context Menus library failed to initialize properly. Right-click might not work as expected.");
        // Append a visual warning if desired
        p.innerHTML += ' <br><span style="color: orange; font-weight: bold;">Warning: Right-click menus might be disabled due to a loading error.</span>';
    }
     console.log("Updated editor placeholder text with detailed instructions.");
}
function addNodeFromMenu(nodeType, position) { if (!cy) return; newNodeCounter++; const idBase = nodeType === 'input' ? 'Input' : 'Hidden'; let id = `${idBase}_${newNodeCounter}`; while (cy.getElementById(id).length > 0) { newNodeCounter++; id = `${idBase}_${newNodeCounter}`; } const name = prompt(`Enter display name for new ${nodeType} node (ID: ${id}):`, id); if (name === null) return; const newNodeData = { group: 'nodes', data: { id: id.trim(), fullName: name.trim() || id.trim(), nodeType: nodeType }, position: position }; cy.add(newNodeData); console.log(`Added ${nodeType} node: ID=${id}, Name=${newNodeData.data.fullName}`); if (nodeType === 'input') { updateInputControls(cy.nodes().map(n => n.data())); } updateNodeProbabilities({}); runLayout(); markConfigUnsaved(); }
function deleteElement(target) { if (!cy || !target || (!target.isNode && !target.isEdge)) return; const id = target.id(); const type = target.isNode() ? 'node' : 'edge'; let name = target.data('fullName') || id; if (target.isEdge()){ name = `${target.source().id()}->${target.target().id()}`; } if (confirm(`Delete ${type} "${name}"?`)) { const wasInputNode = target.isNode() && target.data('nodeType') === 'input'; cy.remove(target); console.log(`Removed ${type}: ${id}`); if(wasInputNode){ updateInputControls(cy.nodes().map(n => n.data())); } markConfigUnsaved(); } }
function convertNodeType(targetNode) { if (!cy || !targetNode || !targetNode.isNode()) return; const currentType = targetNode.data('nodeType'); const newType = currentType === 'input' ? 'hidden' : 'input'; const nodeId = targetNode.id(); if (confirm(`Convert node "${targetNode.data('fullName') || nodeId}" from ${currentType} to ${newType}?`)) { targetNode.data('nodeType', newType); console.log(`Converted node ${nodeId} to ${newType}`); cy.style().update(); updateInputControls(cy.nodes().map(n => n.data())); updateNodeProbabilities({}); markConfigUnsaved(); } }
function startEdgeCreation(sourceNode) { if (!cy || !sourceNode || !sourceNode.isNode()) return; edgeSourceNode = sourceNode; setStatusMessage('config-status', `Edge started from ${sourceNode.id()}. Left-click target node...`, 'loading'); sourceNode.addClass('edge-source-active'); console.log("Edge creation started from:", sourceNode.id()); }
function handleNodeTap(event) { const targetNode = event.target; if (edgeSourceNode && edgeSourceNode.id() !== targetNode.id()) { const sourceId = edgeSourceNode.id(); const targetId = targetNode.id(); if (cy.edges(`[source = "${sourceId}"][target = "${targetId}"]`).length > 0) { setStatusMessage('config-status', `Edge ${sourceId}->${targetId} already exists.`, "error"); } else { const tempEdges = cy.edges().map(e => ({ source: e.source().id(), target: e.target().id() })); tempEdges.push({ source: sourceId, target: targetId }); const tempGraph = { nodes: cy.nodes().map(n => n.data()), edges: tempEdges }; if (!simulateIsDag(tempGraph)) { alert(`Cannot add edge ${sourceId}->${targetId} (creates cycle).`); setStatusMessage('config-status', `Edge aborted (creates cycle).`, "error"); } else { cy.add({ group: 'edges', data: { source: sourceId, target: targetId } }); console.log(`Added edge ${sourceId}->${targetId}`); setStatusMessage('config-status', `Edge ${sourceId}->${targetId} added.`, "success"); markConfigUnsaved(); } } cancelEdgeCreation(); } else if (edgeSourceNode && edgeSourceNode.id() === targetNode.id()) { cancelEdgeCreation(); } }
function handleCanvasTap(event) { if (edgeSourceNode) { cancelEdgeCreation(); } }
function cancelEdgeCreation() { if (edgeSourceNode) { edgeSourceNode.removeClass('edge-source-active'); } edgeSourceNode = null; setTimeout(() => { const currentStatusEl = document.getElementById('config-status'); if(currentStatusEl && (currentStatusEl.textContent.includes('Edge started') || currentStatusEl.textContent.includes('aborted'))) { const currentNameEl = document.getElementById('current-config-name'); const msg = currentConfig ? `Config '${currentNameEl ? currentNameEl.textContent.replace('*','') : '?'}' loaded.` : "Ready."; setStatusMessage('config-status', msg , "success"); } }, 1500); }

async function loadDefaultConfig() { console.log("Fetching default configuration structure..."); try { await retryFetch(async () => { const r = await fetch('/api/configs/default'); if (!r.ok) throw new Error(`HTTP ${r.status}`); defaultGraphStructure = await r.json(); console.log("Default structure stored."); }, 3); } catch (e) { console.error('Error fetching default struct:', e); defaultGraphStructure = null; throw e; } }
function addDefaultToDropdown() { const s=document.getElementById('load-config-select'); if(!defaultGraphStructure || !s)return; let e=false; for(let i=0;i<s.options.length;i++){if(s.options[i].value===defaultGraphStructure.id){e=true;break;}} if(!e){const o=document.createElement('option');o.value=defaultGraphStructure.id;o.textContent=`${defaultGraphStructure.name} (Default)`; if(s.options.length>0&&s.options[0].value===""){s.insertBefore(o,s.options[1]);}else{s.insertBefore(o,s.firstChild);} console.log("Added Default to dropdown.");} }
function loadGraphData(configData, isDefault=false) { if(!cy||!configData||!configData.graph_structure){console.error("Load graph error"); return;} console.log(`Loading: ${configData.name}`); const elems = configData.graph_structure.nodes.map(n=>({data:{id:n.id,fullName:n.fullName,nodeType:n.nodeType}})).concat(configData.graph_structure.edges.map(e=>({data:{source:e.source,target:e.target}}))); cy.elements().remove(); cy.add(elems); runLayout(); currentConfig={...configData}; const configNameInput = document.getElementById('config-name'); if (configNameInput) configNameInput.value=isDefault?'':currentConfig.name; const currentNameEl = document.getElementById('current-config-name'); if(currentNameEl) currentNameEl.textContent=currentConfig.name; updateInputControls(currentConfig.graph_structure.nodes); updateNodeProbabilities({}); clearSessionLog(); clearLLMOutputs(); clearUnsavedMark(); selectConfigInDropdown(currentConfig.id); }
async function saveConfiguration() { const ni=document.getElementById('config-name'); const n=ni.value.trim(); if(!n){setStatusMessage('config-status',"Enter name.","error");return;} if(!cy){setStatusMessage('config-status',"Graph not init.","error");return;} setStatusMessage('config-status',"Saving...","loading"); showSpinner(true); enableUI(false); const struct = {nodes: cy.nodes().map(node=>({id:node.id(),fullName:node.data('fullName'),nodeType:node.data('nodeType')})), edges: cy.edges().map(edge=>({source:edge.source().id(),target:edge.target().id()}))}; await retryFetch(async()=>{ const r=await fetch('/api/configs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config_name:n,graph_structure:struct})}); const d=await r.json(); if(!r.ok)throw new Error(d.detail||`HTTP ${r.status}`); currentConfig={id:d.config_id,name:d.config_name,graph_structure:struct}; const currentNameEl = document.getElementById('current-config-name'); if(currentNameEl) currentNameEl.textContent=currentConfig.name; if(ni) ni.value=''; setStatusMessage('config-status',`Saved as '${d.config_name}'.`,"success"); await loadConfigList(); selectConfigInDropdown(d.config_id); clearSessionLog(); clearUnsavedMark();},3,()=>setStatusMessage('config-status',"Save fail. Retry...","error")).catch(e=>{console.error('Save error:',e); setStatusMessage('config-status',`Save fail: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);}); }
async function loadConfigList() { console.log("Updating saved list..."); try { await retryFetch(async()=>{ const r=await fetch('/api/configs'); if(!r.ok)throw new Error(`HTTP ${r.status}`); const configs=await r.json(); const s=document.getElementById('load-config-select'); const curSel=s.value; const optsToRemove=[]; for(let i=0;i<s.options.length;i++){const v=s.options[i].value; if(v!==""&&v!=="default-config-001"){optsToRemove.push(s.options[i]);}} optsToRemove.forEach(opt=>s.removeChild(opt)); configs.forEach(c=>{const o=document.createElement('option');o.value=c.id;o.textContent=c.name;s.appendChild(o);}); s.value=curSel; let foundCur=false; if(currentConfig){for(let i=0;i<s.options.length;i++){if(s.options[i].value===currentConfig.id){s.value=currentConfig.id;foundCur=true;break;}}} if(!foundCur&&defaultGraphStructure){s.value=defaultGraphStructure.id;} if(s.value==="")s.selectedIndex=0; console.log("Saved list updated.");},3); } catch(e){ console.error('Load list error:',e); setStatusMessage('config-status',`Failed list load: ${e.message}`,"error");}}
async function loadSelectedConfiguration() { const s=document.getElementById('load-config-select'); const id=s.value; if(!id){setStatusMessage('config-status',"Select config.","error");return;} if(!cy){setStatusMessage('config-status',"Graph not ready.","error");return;} setStatusMessage('config-status',`Loading ${s.options[s.selectedIndex].text}...`,"loading"); showSpinner(true); enableUI(false); await retryFetch(async()=>{ if(id==='default-config-001'&&defaultGraphStructure){loadGraphData(defaultGraphStructure,true);setStatusMessage('config-status',"Default loaded.","success");return;} const r=await fetch(`/api/configs/${id}`); if(!r.ok){const e=await r.json().catch(()=>({detail:`HTTP ${r.status}`}));throw new Error(e.detail||`HTTP ${r.status}`);} const d=await r.json(); loadGraphData(d,false); setStatusMessage('config-status',`Config '${d.name}' loaded.`,"success");},3,()=>setStatusMessage('config-status',"Load fail. Retry...","error")).catch(e=>{console.error('Load select error:',e); setStatusMessage('config-status',`Load failed: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);});}
async function setDefaultConfiguration() { const s=document.getElementById('load-config-select'); const id=s.value; const name=s.options[s.selectedIndex].text; if(!id||id==='default-config-001'){setStatusMessage('config-status',"Select saved config.","error");return;} if(!confirm(`Set "${name}" as default?`))return; setStatusMessage('config-status',`Setting default...`,"loading"); showSpinner(true); enableUI(false); await retryFetch(async()=>{ const r=await fetch('/api/configs/set_default',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(id)}); const d=await r.json(); if(!r.ok)throw new Error(d.detail||`HTTP ${r.status}`); setStatusMessage('config-status',`Set '${name}' as default.`,"success"); await loadDefaultConfig();},3,()=>setStatusMessage('config-status',"Set default fail. Retry...","error")).catch(e=>{console.error('Set default error:',e);setStatusMessage('config-status',`Set default failed: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);});}
async function deleteSelectedConfiguration() { const s=document.getElementById('load-config-select'); const id=s.value; const name=s.options[s.selectedIndex].text; if(!id||id==='default-config-001'){setStatusMessage('config-status',"Select saved config.","error");return;} if(!confirm(`Delete "${name}"?`))return; setStatusMessage('config-status',`Deleting...`,"loading"); showSpinner(true); enableUI(false); await retryFetch(async()=>{ const r=await fetch(`/api/configs/${id}`,{method:'DELETE'}); const d=await r.json(); if(!r.ok)throw new Error(d.detail||`HTTP ${r.status}`); setStatusMessage('config-status',`Deleted '${name}'.`,"success"); if(currentConfig && currentConfig.id===id){if(defaultGraphStructure){loadGraphData(defaultGraphStructure,true);setStatusMessage('config-status',`Deleted '${name}'. Default loaded.`,"success");}else{cy.elements().remove();currentConfig=null;updateInputControls([]);clearSessionLog();clearLLMOutputs();document.getElementById('current-config-name').textContent="None";setStatusMessage('config-status',`Deleted '${name}'. Load another.`,"success");}} await loadConfigList();},3,()=>setStatusMessage('config-status',"Delete fail. Retry...","error")).catch(e=>{console.error('Delete error:',e); setStatusMessage('config-status',`Delete failed: ${e.message}`,"error");}).finally(()=>{enableUI(true);showSpinner(false);});}
function initializeCytoscape() {
    console.log("Attempting Cytoscape core initialization...");
    try {
        if (typeof nodeLabelFunc !== 'function') { throw new Error("Internal Error: nodeLabelFunc is not defined before Cytoscape init."); }
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: [],
            style: [ /* styles using nodeLabelFunc */
                 { selector: 'node', style: { 'background-color': '#ccc', 'label': nodeLabelFunc, 'width': 120, 'height': 120, 'shape': 'ellipse', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px', 'font-weight': '100', 'text-wrap': 'wrap', 'text-max-width': 110, 'text-outline-color': '#fff', 'text-outline-width': 1, 'color': '#333', 'transition-property': 'background-color, color', 'transition-duration': '0.5s' } },
                 { selector: 'node[nodeType="input"]', style: { 'shape': 'rectangle', 'width': 130, 'height': 70 } },
                 { selector: 'node[nodeType="hidden"]', style: { 'shape': 'ellipse' } },
                 { selector: 'edge', style: { 'width': 2, 'line-color': '#666', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#666', 'curve-style': 'bezier' } },
                 { selector: '.edge-source-active', style: { 'border-width': 3, 'border-color': '#00ff00' } },
                 { selector: '.eh-grabbed', style: { 'border-width': 3, 'border-color': '#007bff' } }
             ],
            layout: { name: 'preset' }
        });
        console.log("Cytoscape core initialized successfully.");
    } catch(error) {
        console.error("Failed to initialize Cytoscape Core:", error);
        alert(`Critical Error initializing graph library: ${error.message}\n\n(Check console for more details)`);
        cy = null;
    }
}
function initializeEditingExtensions() {
    console.log("Initializing and Registering editing extensions...");
    if (!cy) { console.error("Cannot init extensions: Cytoscape not ready."); return; }

    // --- Register & Initialize Context Menus ---
    try {
        // Check if library AND dependencies are loaded AND registration hasn't happened
        if (typeof cytoscapeContextMenus === 'function' && typeof tippy === 'function' && typeof Popper === 'object' && !didRegisterContextMenus) {
             cytoscape.use( cytoscapeContextMenus ); // REGISTER FIRST
             didRegisterContextMenus = true; // Mark as registered only once
             console.log("Context Menus registered.");
             contextMenuInstance = cy.contextMenus({ // INITIALIZE instance
                 menuItems: [ // Use 'content' for V4.x
                     { id: 'add-input-core', content: 'Add Input Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('input', evt.position || evt.cyPosition) },
                     { id: 'add-hidden-core', content: 'Add Hidden Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('hidden', evt.position || evt.cyPosition), hasTrailingDivider: true },
                     { id: 'start-edge', content: 'Start Edge From Here', selector: 'node', onClickFunction: (evt) => startEdgeCreation(evt.target || evt.cyTarget)}, // ADDED Click-to-connect start
                     { id: 'convert-type', content: 'Convert Type', selector: 'node', onClickFunction: (evt) => convertNodeType(evt.target || evt.cyTarget) },
                     { id: 'delete-node', content: 'Delete Node', selector: 'node', hasTrailingDivider: true, onClickFunction: (evt) => deleteElement(evt.target || evt.cyTarget) },
                     { id: 'delete-edge', content: 'Delete Edge', selector: 'edge', onClickFunction: (evt) => deleteElement(evt.target || evt.cyTarget) }
                 ],
                 menuItemClasses: [ 'ctx-menu-item' ], contextMenuClasses: [ 'ctx-menu' ]
             });
            console.log("Context Menus instance initialized.");
        } else if (!didRegisterContextMenus) { console.warn("cytoscape-context-menus lib or dependencies not found globally."); }
         else { console.log("Context Menus already registered."); }
    } catch (e) { console.error("Error during Context Menus registration/initialization:", e); contextMenuInstance = null; }

     updateEditorPlaceholderText(); // Update message based on final success
}
// function initializeUI() { /* MODIFIED Below */ }

// async function fetchAndUpdateLLM() { /* REMOVED - Replaced by runFullSimulation */ }


// --- NEW/MODIFIED Helper Functions ---

// Dynamically create persona inputs (A1-A6, H)
function updatePersonaInputControls() {
    const container = document.getElementById('persona-input-controls-container');
    if (!container) return;
    container.innerHTML = ''; // Clear existing

    const personaNodes = [ // Define the persona nodes explicitly
        { id: "A1", fullName: "Domain Expertise" },
        { id: "A2", fullName: "Web Literacy" },
        { id: "A3", fullName: "Task Familiarity" },
        { id: "A4", fullName: "Goal Clarity" },
        { id: "A5", fullName: "Cognitive Capacity" },
        { id: "A6", fullName: "Risk Aversion" },
        { id: "H", fullName: "Recent History" },
    ];

    personaNodes.forEach(node => {
        const div = document.createElement('div');
        const label = document.createElement('label');
        label.htmlFor = `input-${node.id}`;
        label.textContent = `${node.id} (${node.fullName}):`;

        const input = document.createElement('input');
        input.type = 'number';
        input.id = `input-${node.id}`; // Use standard prefix
        input.name = node.id;
        input.min = "0";
        input.max = "1";
        input.step = "0.01";
        input.value = "0.5"; // Default value

        div.appendChild(label);
        div.appendChild(input);
        container.appendChild(div);
    });
}

// Display Objective UI Scores
function displayObjectiveUIScores(features) {
    const container = document.getElementById('ui-objective-scores');
    if (!container) return;
    if (!features) {
        container.innerHTML = '<h3>Objective UI Scores (0-1):</h3><p>Waiting for analysis...</p>';
        return;
    }
    let html = '<h3>Objective UI Scores (0-1):</h3><ul>';
    // REMOVED UI_Guidance
    const featureOrder = ["Clarity", "Familiarity", "Findability", "Complexity", "AestheticTrust"];
    featureOrder.forEach(key => {
        if (features.hasOwnProperty(key)) {
            html += `<li><strong>${key}:</strong> ${features[key].toFixed(3)}</li>`;
        }
    });
    html += '</ul>';
    container.innerHTML = html;
}

// Display User Perception Summary
function displayUserPerception(summary) {
     const container = document.getElementById('user-perception-summary');
     if (!container) return;
     container.innerHTML = `<p>${summary || "Run full simulation after UI Analysis and setting Persona Inputs..."}</p>`;
}

// Display Debug Context (Optional)
function displayDebugContext(context) {
    const container = document.getElementById('debug-context');
    if (!container || !context) {
        if(container) container.innerHTML = '<p>Run simulation to see debug context.</p>';
        return;
    }
    let html = '<h4>Inputs Sent to Reasoning:</h4><pre>';
    html += `Persona:\n${JSON.stringify(context.persona_inputs, null, 2)}\n`;
    html += `Cognitive States:\n${JSON.stringify(context.cognitive_states_est, null, 2)}\n`;
    html += `Outcomes:\n${JSON.stringify(context.outcomes_calc, null, 2)}`;
    html += '</pre>';
    container.innerHTML = html;
}

// Clear all dynamic outputs
function clearLLMOutputs() {
    const reasonEl=document.getElementById('llm-reasoning-content'); if(reasonEl) reasonEl.textContent='Run simulation...';
    const perceptionEl = document.getElementById('user-perception-summary'); if(perceptionEl) perceptionEl.innerHTML = '<p>Run simulation...</p>';
    const scoresEl = document.getElementById('ui-objective-scores'); if(scoresEl) scoresEl.innerHTML = '<h3>Objective UI Scores (0-1):</h3><p>Run UI Analysis...</p>';
    const debugEl = document.getElementById('debug-context'); if(debugEl) debugEl.innerHTML = '<p>Run simulation...</p>';
    setStatusMessage('predict-status', "", "");
    setStatusMessage('ui-analysis-status', "", "");
}

// --- Log Prediction (MODIFIED for new structure) ---
function logPrediction(inputs, uiFeatures, nodeProbs, perception, reasoning) {
    const timestamp = new Date().toISOString();

    // Ensure nodeProbs is the flattened P(1) map
    const flatNodeProbs = {};
     for (const nodeId in nodeProbs) {
         if (nodeProbs[nodeId] && typeof nodeProbs[nodeId]["1"] === 'number') {
             flatNodeProbs[nodeId] = nodeProbs[nodeId]["1"];
         }
     }
     // Also add persona inputs to the node probability map for logging completeness
     for (const inputId in inputs) {
         flatNodeProbs[inputId] = inputs[inputId];
     }

    const logEntry = {
        timestamp: timestamp,
        configId: currentConfig ? currentConfig.id : "unsaved",
        configName: currentConfig ? currentConfig.name : "Unsaved",
        uiFeatures: { ...(uiFeatures || {}) }, // Include UI features used
        nodeProbabilities: flatNodeProbs,    // Include all node P(1) values
        userPerceptionSummary: perception || "", // Include perception text
        llmReasoning: reasoning || ""       // Include reasoning text
    };

    sessionLog.push(logEntry);
    const logCountEl = document.getElementById('log-count');
    if (logCountEl) logCountEl.textContent = `Session logs: ${sessionLog.length}`;
    console.log("Prediction added to session log (incl. UI/Text):", logEntry);
}

// --- Download Session Log (MODIFIED for new structure) ---
function downloadSessionLog() {
    if (sessionLog.length === 0) { alert("No logs."); return; }
    console.log("Generating session log CSV (row per prediction)...");

    // --- Determine Dynamic Headers ---
    const allNodeIds = new Set();
    const allFeatureIds = new Set();
    sessionLog.forEach(entry => {
        Object.keys(entry.nodeProbabilities || {}).forEach(nodeId => allNodeIds.add(nodeId));
        Object.keys(entry.uiFeatures || {}).forEach(featId => allFeatureIds.add(featId));
    });
    const sortedNodeIds = Array.from(allNodeIds).sort();
    const sortedFeatureIds = Array.from(allFeatureIds).sort();

    // Define the final headers including new text fields
    const csvHeaders = ["Timestamp", "ConfigID", "ConfigName",
                        ...sortedFeatureIds, ...sortedNodeIds,
                        "UserPerceptionSummary", "LLMReasoning"];
    console.log("CSV Headers:", csvHeaders);

    // --- Map Log Data to Rows ---
    const csvData = sessionLog.map(entry => {
        const row = {
            Timestamp: entry.timestamp,
            ConfigID: entry.configId,
            ConfigName: entry.configName,
        };
        // Add UI Feature values
        sortedFeatureIds.forEach(featId => {
            const val = entry.uiFeatures ? entry.uiFeatures[featId] : undefined;
            row[featId] = (val !== undefined && typeof val === 'number') ? val.toFixed(4) : '';
        });
        // Add Node Probability P(1) values
        sortedNodeIds.forEach(nodeId => {
            const probValue = entry.nodeProbabilities ? entry.nodeProbabilities[nodeId] : undefined;
            row[nodeId] = (probValue !== undefined && typeof probValue === 'number') ? probValue.toFixed(4) : '';
        });
        // Add Text fields (PapaParse handles basic escaping)
        row["UserPerceptionSummary"] = entry.userPerceptionSummary || "";
        row["LLMReasoning"] = entry.llmReasoning || "";
        return row;
    });

    // --- Generate and Download CSV ---
    try {
        const csvString = Papa.unparse({ fields: csvHeaders, data: csvData });
        triggerCsvDownload(csvString, `session_log_${(currentConfig?.name || 'unsaved').replace(/[^a-z0-9]/gi, '_')}`);
        console.log("Session log CSV download triggered.");
    } catch (error) { console.error("Error generating session CSV:", error); alert(`Error generating CSV: ${error.message}`); }
}


// --- NEW: Analyze UI Function ---
async function analyzeUI() {
    const imageInput = document.getElementById('ui-image-upload');
    const taskInput = document.getElementById('task-description');
    const statusEl = document.getElementById('ui-analysis-status');

    if (!imageInput || !imageInput.files || imageInput.files.length === 0) {
        setStatusMessage('ui-analysis-status', "Please select a UI screenshot image.", "error");
        return;
    }
    const taskDescription = taskInput ? taskInput.value.trim() : "";
    if (!taskDescription) {
        setStatusMessage('ui-analysis-status', "Please enter a task description.", "error");
        return;
    }

    const file = imageInput.files[0];
    // Basic type check (sync)
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
         setStatusMessage('ui-analysis-status', "Invalid file type (PNG, JPG, WEBP only).", "error");
         return;
    }

    setStatusMessage('ui-analysis-status', "Analyzing UI...", "loading");
    showSpinner(true);
    enableUI(false);

    try {
        const formData = new FormData();
        formData.append("image", file);
        formData.append("task_description", taskDescription);

        // Use retryFetch for the API call
        let analysisResult;
        await retryFetch(async () => {
            const response = await fetch('/api/extract_ui_features', {
                method: 'POST',
                body: formData, // Send as FormData
            });
            analysisResult = await response.json(); // Expect JSON response
            if (!response.ok) {
                throw new Error(analysisResult.detail || `HTTP error ${response.status}`);
            }
        }, 3, () => setStatusMessage('ui-analysis-status', "Analysis failed. Retrying...", "error"));

        currentUiFeatures = analysisResult; // Store the result globally
        displayObjectiveUIScores(currentUiFeatures); // Display the scores
        setStatusMessage('ui-analysis-status', "UI Analysis complete. Ready to run simulation.", "success");
        console.log("UI Features:", currentUiFeatures);

    } catch (error) {
        console.error("UI Analysis Error:", error);
        setStatusMessage('ui-analysis-status', `Analysis failed: ${error.message}`, "error");
        currentUiFeatures = null; // Reset on error
        displayObjectiveUIScores(null);
    } finally {
        enableUI(true);
        showSpinner(false);
    }
}

// --- NEW: Run Full Simulation Function ---
async function runFullSimulation() {
    // 1. Check prerequisites
    if (!currentConfig || !currentConfig.graph_structure || !currentConfig.graph_structure.nodes.length === 0) { alert("Load or define a graph configuration first."); return; }
    if (!cy) { alert("Graph library not ready."); return; }
    if (!currentUiFeatures) { alert("Please analyze a UI image first using the 'Analyze UI' button."); return; }

    setStatusMessage('predict-status', "Gathering persona inputs...", "loading");

    // 2. Gather Persona Inputs
    const personaInputValues = {};
    const personaNodes = ["A1", "A2", "A3", "A4", "A5", "A6", "H"]; // Expected persona node IDs
    let hasInvalidInput = false;
    personaNodes.forEach(nodeId => {
        const inputElement = document.getElementById(`input-${nodeId}`);
        const inputContainer = inputElement?.parentElement;
        if (inputElement) {
            const value = parseFloat(inputElement.value);
            if (isNaN(value) || value < 0 || value > 1) {
                 setStatusMessage('predict-status', `Invalid input for ${nodeId}. Use 0-1.`, "error");
                 if(inputContainer) inputContainer.classList.add('invalid-input');
                 hasInvalidInput = true;
             } else {
                 if(inputContainer) inputContainer.classList.remove('invalid-input');
                 personaInputValues[nodeId] = value;
             }
        } else {
             console.warn(`Input element for persona node ${nodeId} not found. Using default 0.5`);
             personaInputValues[nodeId] = 0.5; // Default if element is missing (e.g., 'H' if optional)
        }
    });
    if(hasInvalidInput) { return; } // Stop if input is invalid

    // 3. Prepare Payload for the *new* backend endpoint
    const payload = {
        persona_inputs: personaInputValues, // Send the collected values
        ui_features: currentUiFeatures,    // Send the results from UI analysis
        graph_structure: currentConfig.graph_structure,
        config_id: currentConfig.id,
        config_name: currentConfig.name
    };

    // 4. Call Backend for Full Simulation
    setStatusMessage('predict-status', "Running full simulation (Stages 2-4)...", "loading");
    showSpinner(true);
    enableUI(false);
    clearLLMOutputs(); // Clear previous reasoning/perception display

    try {
        let simulationResult;
        await retryFetch(async () => {
            const response = await fetch('/api/predict_full_simulation', { // Call the NEW endpoint
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            simulationResult = await response.json();
            if (!response.ok) {
                throw new Error(simulationResult.detail || `HTTP error ${response.status}`);
            }
        }, 3, () => setStatusMessage('predict-status', "Simulation failed. Retrying...", "error"));

        // 5. Update UI with results
        updateNodeProbabilities(simulationResult.probabilities);    // Update graph nodes
        displayUserPerception(simulationResult.user_perception_summary); // Display perception text
        displayLLMReasoning(simulationResult.llm_reasoning);        // Display reasoning text
        displayDebugContext(simulationResult.debug_context);        // Display debug info
        setStatusMessage('predict-status', "Simulation complete.", "success");

        // 6. Log the results (including UI features, perception, reasoning)
        logPrediction(
            personaInputValues,
            simulationResult.ui_features, // Pass UI features
            simulationResult.probabilities, // Pass node probabilities (already flattened in backend ideally, but handle here if needed)
            simulationResult.user_perception_summary,
            simulationResult.llm_reasoning
        );

    } catch (error) {
        console.error("Full Simulation Error:", error);
        setStatusMessage('predict-status', `Simulation failed: ${error.message}`, "error");
        clearLLMOutputs(); // Clear outputs on error
    } finally {
        enableUI(true);
        showSpinner(false);
    }
}


// --- Initialize UI (MODIFIED) ---
function initializeUI() {
    console.log("Initializing UI event listeners...");
    // Config Buttons
    document.getElementById('save-config-button')?.addEventListener('click', saveConfiguration);
    document.getElementById('load-config-button')?.addEventListener('click', loadSelectedConfiguration);
    document.getElementById('set-default-button')?.addEventListener('click', setDefaultConfiguration);
    document.getElementById('delete-config-button')?.addEventListener('click', deleteSelectedConfiguration);

    // NEW UI Analysis Button
    document.getElementById('analyze-ui-button')?.addEventListener('click', analyzeUI);

    // Simulation Button (replaces old update button)
    document.getElementById('run-simulation-button')?.addEventListener('click', runFullSimulation);

    // Other Controls
    document.getElementById('gradient-toggle')?.addEventListener('change', () => updateNodeProbabilities(null)); // Re-render colors
    document.getElementById('download-session-log-button')?.addEventListener('click', downloadSessionLog);
    document.getElementById('download-all-log-button')?.addEventListener('click', downloadAllLogs);
    console.log("UI Listeners attached.");

    // Initialize Persona Inputs Display
    updatePersonaInputControls();
    // Initialize Objective Scores Display
    displayObjectiveUIScores(null);
    // Initialize Perception Summary Display
    displayUserPerception(null);
    // Initialize Debug Display
    displayDebugContext(null);

}


// --- Cytoscape Event Listeners Setup Function ---
function setupCytoscapeEventListeners() {
    // ... (Keep as before: adds tap listeners for click-to-connect) ...
    if (!cy) { console.error("Cannot setup listeners, Cytoscape not initialized."); return; }
    cy.off('tap', 'node', handleNodeTap); cy.off('tap', handleCanvasTap); // Remove old before adding
    cy.on('tap', 'node', handleNodeTap); cy.on('tap', handleCanvasTap);
    console.log("Cytoscape tap listeners added/updated for edge creation.");
}


// ==================================================
// --- DOMContentLoaded Listener (AT THE END) ---
// ==================================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log("DOM Content Loaded. Starting initialization sequence...");

    if (typeof cytoscape !== 'function') { alert("Error: Cytoscape library failed to load."); setStatusMessage('config-status', "Error: Core graph library failed.", "error"); showSpinner(false); return; }
    if (!document.getElementById('cy')) { alert("Error: Graph container element 'cy' not found."); setStatusMessage('config-status', "Error: Graph container missing.", "error"); showSpinner(false); return; }

    initializeCytoscape(); // 1. Create 'cy' instance

    if (!cy) { showSpinner(false); return; } // Stop if core failed

    initializeEditingExtensions(); // 2. Try to register and init editing extensions

    initializeUI(); // 3. Set up button listeners & Initial UI state

    setupCytoscapeEventListeners(); // 4. Add tap listeners for edge creation

    // 5. Load Data
    showSpinner(true);
    setStatusMessage('config-status', "Loading config data...", "loading");
    try {
        await loadDefaultConfig();
        if (defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true);
            addDefaultToDropdown();
            const select = document.getElementById('load-config-select');
            if(!select || !select.value || select.value === "") { // Added check for select existence
                 selectConfigInDropdown(defaultGraphStructure.id);
            }
        } else { throw new Error("Default config data unavailable."); }
        await loadConfigList();
        const finalStatus = currentConfig ? `Config '${currentConfig.name}' loaded.` : "Ready.";
        if (!document.getElementById('config-status').classList.contains('error')){ setStatusMessage('config-status', finalStatus, "success"); }
    } catch (error) {
        console.error("Initialization Data Load Error:", error);
        if (!document.getElementById('config-status').classList.contains('error')) { setStatusMessage('config-status', `Initialization failed: ${error.message}`, "error"); }
        if (!currentConfig) { updatePersonaInputControls(); /* Update inputs even on error */ const currentNameEl = document.getElementById('current-config-name'); if(currentNameEl) currentNameEl.textContent = "None"; }
    } finally {
        showSpinner(false);
    }
});

console.log("script.js loaded and parsed. Waiting for DOMContentLoaded.");
