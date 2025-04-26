// public/script.js

// --- Global Variables ---
let cy;
let currentConfig = null;
let defaultGraphStructure = null;
let sessionLog = [];
let edgeHandlesInstance = null;
let contextMenuInstance = null;
let newNodeCounter = 0;
let didRegisterEdgehandles = false;
let didRegisterContextMenus = false;

// ===========================================
// --- ALL HELPER FUNCTION DEFINITIONS ---
// ===========================================

function nodeLabelFunc(node) { /* ... as defined before ... */ }
function simulateIsDag(graph) { /* ... as defined before ... */ }
function selectConfigInDropdown(configId) { /* ... as defined before ... */ }
function enableUI(enable) { /* ... as defined before ... */ }
function showSpinner(show) { /* ... as defined before ... */ }
function runLayout() { /* ... as defined before ... */ }
function updateInputControls(nodes) { /* ... as defined before ... */ }
function updateNodeProbabilities(probabilities) { /* ... as defined before ... */ }
function displayLLMReasoning(reasoningText) { /* ... as defined before ... */ }
function displayLLMContext(context) { /* ... as defined before ... */ }
function setStatusMessage(elementId, message, type) { /* ... as defined before ... */ }
async function retryFetch(fetchFn, maxRetries, onRetry) { /* ... as defined before ... */ }
function logPrediction(inputs, probabilities) { /* ... as defined before ... */ }
function clearSessionLog() { /* ... as defined before ... */ }
function clearLLMOutputs() { /* ... as defined before ... */ }
function downloadSessionLog() { /* ... as defined before ... */ }
async function downloadAllLogs() { /* ... as defined before ... */ }
function triggerCsvDownload(csvDataOrBlob, baseFilename) { /* ... as defined before ... */ }
function markConfigUnsaved() { /* ... as defined before ... */ }
function clearUnsavedMark() { /* ... as defined before ... */ }
function updateEditorPlaceholderText() { /* ... as defined before ... */ }
function addNodeFromMenu(nodeType, position) { /* ... as defined before ... */ }
function deleteElement(target) { /* ... as defined before ... */ }
function convertNodeType(targetNode) { /* ... as defined before ... */ }
async function loadDefaultConfig() { /* ... as defined before ... */ }
function addDefaultToDropdown() { /* ... as defined before ... */ }
function loadGraphData(configData, isDefault=false) { /* ... as defined before ... */ }
async function saveConfiguration() { /* ... as defined before ... */ }
async function loadConfigList() { /* ... as defined before ... */ }
async function loadSelectedConfiguration() { /* ... as defined before ... */ }
async function setDefaultConfiguration() { /* ... as defined before ... */ }
async function deleteSelectedConfiguration() { /* ... as defined before ... */ }

// --- Cytoscape Core Initialization ---
function initializeCytoscape() {
    console.log("Attempting Cytoscape core initialization...");
    try {
        if (typeof nodeLabelFunc !== 'function') { throw new Error("Internal Error: nodeLabelFunc is not defined before Cytoscape init."); }
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: [],
            style: [ /* ... your style definitions ... */
                 { selector: 'node', style: { 'background-color': '#ccc', 'label': nodeLabelFunc, 'width': 120, 'height': 120, 'shape': 'ellipse', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px', 'font-weight': '100', 'text-wrap': 'wrap', 'text-max-width': 110, 'text-outline-color': '#fff', 'text-outline-width': 1, 'color': '#333', 'transition-property': 'background-color, color', 'transition-duration': '0.5s' } },
                 { selector: 'node[nodeType="input"]', style: { 'shape': 'rectangle', 'width': 130, 'height': 70 } },
                 { selector: 'node[nodeType="hidden"]', style: { 'shape': 'ellipse' } },
                 { selector: 'edge', style: { 'width': 2, 'line-color': '#666', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#666', 'curve-style': 'bezier' } },
                 { selector: '.edge-source-active', style: { 'border-width': 3, 'border-color': '#00ff00' } },
                 { selector: '.eh-grabbed', style: { 'border-width': 3, 'border-color': '#007bff' } } // Style for grabbed node handle
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

// --- Editing Extensions Initialization & REGISTRATION ---
function initializeEditingExtensions() {
    console.log("Initializing and Registering editing extensions...");
    if (!cy) { console.error("Cannot init extensions: Cytoscape not ready."); return; }

    // --- Register & Initialize Edge Handles ---
    try {
        // Check using the global name exposed by v3.6.0
        if (typeof cytoscapeEdgehandles === 'function' && !didRegisterEdgehandles) {
             cytoscape.use( cytoscapeEdgehandles );
             didRegisterEdgehandles = true; console.log("Edgehandles registered.");
             // No setTimeout needed for v3.6.0 usually
             edgeHandlesInstance = cy.edgehandles({ // INITIALIZE instance for v3.6.0
                  preview: true,      // Show preview line
                  hoverDelay: 150,    // Delay for handle appearance
                  handleNodes: 'node', // Handle appears on nodes
                  snap: true,         // Snap to nodes
                  snapThreshold: 50,  // Snap distance
                  snapFrequency: 15,  // Snap frequency
                  noEdgeEventsInDraw: true, // No events on ghost edge
                  disableBrowserGestures: true, // Prevent scrolling etc.
                  handlePosition: 'middle top',
                  handleSize: 10,
                  handleColor: '#007bff',
                  handleIcon: false,
                  edgeType: (source, target) => 'flat',
                  loopAllowed: (node) => false,
                  nodeLoopOffset: -50,
                  edgeParams: (source, target) => ({ data: { source: source.id(), target: target.id() } }),
                   ghostEdgeParams: () => ({}),
                   complete: ( sourceNode, targetNode, addedEles ) => {
                     console.log("Edge draw completed:", sourceNode.id(), "->", targetNode.id());
                     const currentEdges = cy.edges().map(e=>({source:e.source().id(), target:e.target().id()}));
                     const currentNodes = cy.nodes().map(n=>n.data());
                     const currentGraph = { nodes: currentNodes, edges: currentEdges };
                     if (!simulateIsDag(currentGraph)) {
                         alert(`Cycle detected! Removing edge ${sourceNode.id()}->${targetNode.id()}`);
                         cy.remove(addedEles); // Use addedEles from callback
                         setStatusMessage('config-status', `Edge aborted (cycle).`, "error");
                     } else {
                         markConfigUnsaved();
                         setStatusMessage('config-status', `Edge ${sourceNode.id()}->${targetNode.id()} added.`, "success");
                     }
                   },
                   // CSS Classes (check if needed/different for v3.6.0 if styling breaks)
                   handleClass: 'eh-handle', hoverClass: 'eh-hover', //... etc.
             });
            console.log("Edgehandles instance initialized.");
        } else if (!didRegisterEdgehandles) { console.warn("cytoscape-edgehandles lib function not found globally."); }
        else { console.log("Edgehandles already registered."); }
    } catch (e) { console.error("Error during Edgehandles registration/initialization:", e); edgeHandlesInstance = null; }

    // --- Register & Initialize Context Menus ---
    try {
        if (typeof cytoscapeContextMenus === 'function' && typeof tippy === 'function' && typeof Popper === 'object' && !didRegisterContextMenus) {
             cytoscape.use( cytoscapeContextMenus );
             didRegisterContextMenus = true; console.log("Context Menus registered.");
             contextMenuInstance = cy.contextMenus({
                 menuItems: [ // Using 'content' for v4.x
                     { id: 'add-input-core', content: 'Add Input Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('input', evt.position || evt.cyPosition) },
                     { id: 'add-hidden-core', content: 'Add Hidden Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('hidden', evt.position || evt.cyPosition), hasTrailingDivider: true },
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

// ==================================================
// --- MOVED DOMContentLoaded Listener to the END ---
// ==================================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log("DOM Content Loaded. Starting initialization...");

    if (typeof cytoscape !== 'function') { alert("Error: Cytoscape library failed to load."); setStatusMessage('config-status', "Error: Core graph library failed.", "error"); showSpinner(false); return; }
    if (!document.getElementById('cy')) { alert("Error: Graph container element 'cy' not found."); setStatusMessage('config-status', "Error: Graph container missing.", "error"); showSpinner(false); return; }

    initializeCytoscape(); // 1. Create 'cy' instance

    if (!cy) { showSpinner(false); return; } // Stop if core failed

    initializeEditingExtensions(); // 2. Try to register and init extensions

    initializeUI(); // 3. Set up button listeners

    // 4. Load Data
    showSpinner(true);
    setStatusMessage('config-status', "Loading config data...", "loading");
    try {
        await loadDefaultConfig();
        if (defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true);
            addDefaultToDropdown();
            const select = document.getElementById('load-config-select');
            if(!select.value || select.value === "") { selectConfigInDropdown(defaultGraphStructure.id); }
        } else { throw new Error("Default config data unavailable."); }
        await loadConfigList();
        const finalStatus = currentConfig ? `Config '${currentConfig.name}' loaded.` : "Ready.";
        if (!document.getElementById('config-status').classList.contains('error')){ setStatusMessage('config-status', finalStatus, "success"); }
    } catch (error) {
        console.error("Initialization Data Load Error:", error);
        if (!document.getElementById('config-status').classList.contains('error')) { setStatusMessage('config-status', `Initialization failed: ${error.message}`, "error"); }
        if (!currentConfig) { updateInputControls([]); const currentNameEl = document.getElementById('current-config-name'); if(currentNameEl) currentNameEl.textContent = "None"; }
    } finally {
        showSpinner(false);
    }
});

console.log("script.js loaded and parsed. Waiting for DOMContentLoaded."); // New log message
