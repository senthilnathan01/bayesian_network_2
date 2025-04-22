// --- public/script.js ---

// --- Initialize Cytoscape (cy variable) ---
// Define the graph elements globally or fetch them (here defined manually to match backend)
const graphElements = [
    // Nodes (Inputs) - Add 'label' in data for display
    { data: { id: 'A1', fullName: 'Domain Expertise' } },
    { data: { id: 'A2', fullName: 'Web Literacy' } },
    { data: { id: 'A3', fullName: 'Task Familiarity' } },
    { data: { id: 'A4', fullName: 'Goal Clarity' } },
    { data: { id: 'A5', fullName: 'Motivation' } },
    { data: { id: 'UI', fullName: 'UI State (Quality/Clarity)' } },
    { data: { id: 'H', fullName: 'History (Relevant past interactions)' } },

    // Nodes (Intermediate States)
    { data: { id: 'IS1', fullName: 'Confidence' } },
    { data: { id: 'IS2', fullName: 'Cognitive Load' } },
    { data: { id: 'IS3', fullName: 'Task Understanding' } },
    { data: { id: 'IS4', fullName: 'Interaction Fluency' } },
    { data: { id: 'IS5', fullName: 'Relevant Knowledge Activation' } },

    // Nodes (Outputs)
    { data: { id: 'O1', fullName: 'Predicted Success Probability' } },
    { data: { id: 'O2', fullName: 'Action Speed/Efficiency' } },
    { data: { id: 'O3', fullName: 'Help Seeking Likelihood' } },

    // Edges (Based on the new NODE_PARENTS structure from main.py)
    // IS3 <- A1, A3, A4, H
    { data: { source: 'A1', target: 'IS3' } },
    { data: { source: 'A3', target: 'IS3' } },
    { data: { source: 'A4', target: 'IS3' } },
    { data: { source: 'H', target: 'IS3' } },

    // IS4 <- A2, UI, H
    { data: { source: 'A2', target: 'IS4' } },
    { data: { source: 'UI', target: 'IS4' } },
    { data: { source: 'H', target: 'IS4' } },

    // IS5 <- A1, A3, IS3
    { data: { source: 'A1', target: 'IS5' } },
    { data: { source: 'A3', target: 'IS5' } },
    { data: { source: 'IS3', target: 'IS5' } },

    // IS2 <- IS4, IS3
    { data: { source: 'IS4', target: 'IS2' } },
    { data: { source: 'IS3', target: 'IS2' } },

    // IS1 <- A5, IS2, IS3, IS4, IS5
    { data: { source: 'A5', target: 'IS1' } },
    { data: { source: 'IS2', target: 'IS1' } }, // Note: Qualitative rule says negative influence, but structure is just dependency
    { data: { source: 'IS3', target: 'IS1' } }, // Note: Qualitative rule says negative influence
    { data: { source: 'IS4', target: 'IS1' } },
    { data: { source: 'IS5', target: 'IS1' } },

    // O1 <- IS1, IS2, IS3, IS5
    { data: { source: 'IS1', target: 'O1' } },
    { data: { source: 'IS2', target: 'O1' } }, // Note: Qualitative rule says negative influence
    { data: { source: 'IS3', target: 'O1' } }, // Note: Qualitative rule says negative influence
    { data: { source: 'IS5', target: 'O1' } },

    // O2 <- IS1, IS2, A5
    { data: { source: 'IS1', target: 'O2' } },
    { data: { source: 'IS2', target: 'O2' } }, // Note: Qualitative rule says negative influence
    { data: { source: 'A5', target: 'O2' } },

    // O3 <- IS1, IS2, IS3
    { data: { source: 'IS1', target: 'O3' } }, // Note: Qualitative rule says negative influence
    { data: { source: 'IS2', target: 'O3' } }, // Note: Qualitative rule says positive influence
    { data: { source: 'IS3', target: 'O3' } }  // Note: Qualitative rule says positive influence
];

const cy = cytoscape({
    container: document.getElementById('cy'), // The HTML element to render in

    elements: graphElements, // Use the defined elements

    // --- Define Styles ---
    style: [
        // Style for all nodes
        {
            selector: 'node',
            style: {
                'background-color': '#ccc', // Default color
                // Use a compound label: ID + newline + Full Name + newline + Probability
                'label': function(node) {
                     const id = node.data('id');
                     const fullName = node.data('fullName') || id;
                     const currentLabel = node.data('currentProbLabel'); // Get the calculated probability label part
                     return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`;
                },
                'width': 120,
                'height': 120,
                'shape': 'ellipse',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '10px',
                'text-wrap': 'wrap',
                'text-max-width': 110, // Max width before wrapping label
                'text-outline-color': '#fff', // Outline for readability on colored backgrounds
                'text-outline-width': 1,
                'color': '#333', // Default text color
                'transition-property': 'background-color', // Animate color changes
                'transition-duration': '0.5s'
            }
        },
        // Style for input nodes (optional different styling)
        {
            selector: 'node[id = "A1"], node[id = "A2"], node[id = "A3"], node[id = "A4"], node[id = "A5"], node[id = "UI"], node[id = "H"]',
             style: {
                 'background-color': '#add8e6', // Light blue for inputs
                 'shape': 'rectangle', // Inputs could be rectangles
                 'width': 130,
                 'height': 70
             }
        },
        // Style for output nodes (optional different styling)
        {
            selector: 'node[id = "O1"], node[id = "O2"], node[id = "O3"]',
             style: {
                 'background-color': '#90ee90', // Light green for outputs
                 'shape': 'roundrectangle', // Outputs could be rounded rectangles
                 'width': 130,
                 'height': 70
             }
        },
        // Style for intermediate nodes (IS nodes)
         {
             selector: 'node[id ^= "IS"]', // Selects nodes whose ID starts with "IS"
             style: {
                'background-color': '#ffb6c1', // Light pink for intermediate nodes
                'shape': 'ellipse'
             }
         },
        // Style for all edges
        {
            selector: 'edge',
            style: {
                'width': 2,
                'line-color': '#666',
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#666',
                'curve-style': 'bezier'
            }
        }
    ],

    // --- Define Layout ---
    layout: {
        name: 'cola', // Use Cola for better DAG layout
        animate: true, // Animate layout changes
        ungrabifyWhileSimulating: false, // Allows interaction during layout
        nodeSpacing: 40,
        edgeLength: 150,
        padding: 20,
        randomize: false // Keep layout consistent
    }
});

// --- Function to update node appearance based on probabilities ---
function updateNodeProbabilities(probabilities) {
    cy.nodes().forEach(node => {
        const nodeId = node.id();
        if (probabilities[nodeId] && probabilities[nodeId]["1"] !== undefined) {
            const probState1 = probabilities[nodeId]["1"]; // P(Node=1)

            // Store the probability part of the label separately
            node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`);

            // Update color based on probability 0->1 (blue->red gradient)
            const color = `rgb(${Math.round(255 * probState1)}, ${Math.round(255 * (1 - probState1))}, 0)`; // Red-Green gradient (adjust if blue-red preferred)
            // const color = `rgb(${Math.round(255 * probState1)}, 0, ${Math.round(255 * (1 - probState1))})`; // Red-Blue gradient
            node.style('background-color', color);

        } else {
             // Apply default style and label if no probability data yet
             node.data('currentProbLabel', '(N/A)');
             // Reset color based on node type (Input, Intermediate, Output)
             if (['A1', 'A2', 'A3', 'A4', 'A5', 'UI', 'H'].includes(nodeId)) {
                 node.style('background-color', '#add8e6');
             } else if (['O1', 'O2', 'O3'].includes(nodeId)) {
                  node.style('background-color', '#90ee90');
             } else { // IS nodes
                 node.style('background-color', '#ffb6c1');
             }
        }
         // Re-render the label using the updated data field
         node.style('label', node.data('label')(node)); // Trigger label update by calling the style function
    });

    // Consider re-running layout if node size/shape changes significantly, but probably not needed just for label/color
    // cy.layout({ name: 'cola', animate: true, ungrabifyWhileSimulating: false, nodeSpacing: 40, edgeLength: 150, padding: 20, randomize: false }).run();
}

// --- Function to display LLM context ---
function displayLLMContext(context) {
    const inputDiv = document.getElementById('input-context');
    const structureDiv = document.getElementById('structure-context');
    const rulesDiv = document.getElementById('rules-context');
    const descriptionsDiv = document.getElementById('node-descriptions-context');

    // Display Input States
    let inputHtml = '<h3>Input States Provided:</h3><ul>';
    context.input_states.forEach(input => {
        inputHtml += `<li><strong>${input.node}</strong> (${input.description}): ${input.state} (prob approx ${input.value.toFixed(2)})</li>`;
    });
    inputHtml += '</ul>';
    inputDiv.innerHTML = inputHtml;

    // Display Node Descriptions
     let descHtml = '<h3>Node Descriptions:</h3><ul>';
     Object.entries(context.node_descriptions).forEach(([node, desc]) => {
         descHtml += `<li><strong>${node}</strong>: ${desc}</li>`;
     });
     descHtml += '</ul>';
     descriptionsDiv.innerHTML = descHtml;


    // Display Structure (Dependencies)
    let structureHtml = '<h3>Node Dependencies (DAG):</h3><pre>';
    Object.entries(context.node_dependencies).forEach(([node, parents]) => {
        structureHtml += `${node} <- ${parents.join(', ')}\n`;
    });
    structureHtml += '</pre>';
    structureDiv.innerHTML = structureHtml;

    // Display Qualitative Rules
    let rulesHtml = '<h3>Qualitative Rules Provided:</h3><ul>';
     context.qualitative_rules.forEach(rule => {
         rulesHtml += `<li><strong>${rule.node}</strong> (${rule.description}): ${rule.qualitative || 'No specific rule provided.'}</li>`;
     });
     rulesHtml += '</ul>';
     rulesDiv.innerHTML = rulesHtml;
}


// --- Function to gather inputs and fetch predictions from backend ---
async function fetchAndUpdateLLM() {
    // Get input values (ensure they are floats)
    const inputData = {
        A1: parseFloat(document.getElementById('input-A1').value) || 0.5,
        A2: parseFloat(document.getElementById('input-A2').value) || 0.5,
        A3: parseFloat(document.getElementById('input-A3').value) || 0.5,
        A4: parseFloat(document.getElementById('input-A4').value) || 0.5,
        A5: parseFloat(document.getElementById('input-A5').value) || 0.5,
        UI: parseFloat(document.getElementById('input-UI').value) || 0.5,
        H: parseFloat(document.getElementById('input-H').value) || 0.5,
    };

    // Add loading indicator
    const updateButton = document.getElementById('update-button');
    updateButton.disabled = true;
    updateButton.textContent = 'Updating...';
    document.body.style.cursor = 'wait'; // Change cursor

    // Clear previous LLM Context display
     document.getElementById('input-context').innerHTML = '<h3>Input States Provided:</h3><p>Updating...</p>';
     document.getElementById('structure-context').innerHTML = '<h3>Node Dependencies (DAG):</h3><p>Updating...</p>';
     document.getElementById('rules-context').innerHTML = '<h3>Qualitative Rules Provided:</h3><p>Updating...</p>';
     document.getElementById('node-descriptions-context').innerHTML = '<h3>Node Descriptions:</h3><p>Updating...</p>';


    try {
        // Fetch from the single-call endpoint defined in api/main.py
        // Use a relative path assuming the API is hosted on the same domain under /api
        const response = await fetch('/api/predict_openai_bn_single_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });

        if (!response.ok) {
             // Try to parse error detail from backend, default if parsing fails
             let errorDetail = `HTTP error! status: ${response.status}`;
             try {
                 const errorData = await response.json();
                 errorDetail += `, detail: ${errorData.detail || JSON.stringify(errorData)}`;
             } catch (e) {
                 // If response wasn't JSON (like the HTML error page from a server error)
                 errorDetail += `, response text: ${await response.text()}`;
             }
             throw new Error(errorDetail);
        }

        const result = await response.json();
        const allNodeProbabilities = result.probabilities;
        const llmContext = result.llm_context;

        // Update Cytoscape graph
        updateNodeProbabilities(allNodeProbabilities);

        // Display LLM context
        displayLLMContext(llmContext);


    } catch (error) {
        console.error('Error fetching LLM predictions:', error);
        alert(`Error fetching predictions:\n${error.message}`);
        // Clear LLM Context on error or show error there
        document.getElementById('input-context').innerHTML = '<h3>Input States Provided:</h3><p style="color:red;">Error loading context.</p>';
        document.getElementById('structure-context').innerHTML = '';
        document.getElementById('rules-context').innerHTML = '';
        document.getElementById('node-descriptions-context').innerHTML = '';
    } finally {
        // Remove loading indicator
        updateButton.disabled = false;
        updateButton.textContent = 'Update Probabilities';
        document.body.style.cursor = 'default'; // Restore cursor
    }
}

// --- Add event listener to the button ---
document.getElementById('update-button').addEventListener('click', fetchAndUpdateLLM);

// --- Initialize node appearance and layout on page load ---
// Use a timeout to ensure the container div is ready
setTimeout(() => {
    // Run initial layout
    cy.layout({ name: 'cola', animate: true, ungrabifyWhileSimulating: false, nodeSpacing: 40, edgeLength: 150, padding: 20, randomize: false }).run();
    // Then apply initial styles/labels (N/A)
    updateNodeProbabilities({});
     // Populate initial context section with loading state
     displayLLMContext({
        input_states: [{ node: 'Loading...', description: '', value: 0, state: '' }],
        node_dependencies: { 'Loading...': [] },
        qualitative_rules: [{ node: 'Loading...', description: '', qualitative: '' }],
        node_descriptions: { 'Loading...': 'Loading...' }
     });
}, 100); // Small delay
