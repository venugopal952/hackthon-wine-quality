
document.getElementById('file').addEventListener('change', function(event) {
    const fileInput = event.target;
    const fileName = fileInput.files[0]?.name || '';
    if (fileName) {
        document.getElementById('uploadedFileName').innerText = "Uploaded file: " + fileName;
    }
});

document.getElementById('uploadForm').onsubmit = async function(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.error) {
            alert(data.error);
            return;
        }
        plotTables(data);
        plots();
    } catch (error) {
        alert('Failed to upload or process the file.');
        console.error('Error:', error);
    }
};
function plots(){
    const ts = new Date().getTime();
    document.getElementById('hist-container').innerHTML = `<img src="/static/plots/histogram.png?t=${ts}" alt="hist pic" />`;
    document.getElementById('pca-container').innerHTML = `<img src="/static/plots/pca.png?t=${ts}" alt="pca pic" />`;
}

function plotTables(data){
        const columns = Object.keys(data[0]);
        let anomalyCount = 0, normalCount = 0;

        // Separate tables
        let normalHtml = '<div style="padding-left: 10px;"> <h2>Normal records </h2></div> <table border="1"><tr>';
        let anomalyHtml = '</br><div style="padding-left: 10px;"> <h2>Anomaly records </h2></div> <table border="1"><tr>';
        columns.forEach(col => {
            if(col!="anomaly_result"){
                normalHtml += '<th>' + col + '</th>';
                anomalyHtml += '<th>' + col + '</th>';
            }
        });
        normalHtml += '</tr>';
        anomalyHtml += '</tr>';

        // Separate rows
        data.forEach(row => {
            const isAnomaly = row.anomaly_result === 'Anomaly';
            if (isAnomaly) anomalyCount++; else normalCount++;

            let rowHtml = '<tr>';
            columns.forEach(col => { 
                if(col=="anomaly_result"){
                    //do nothing - skip
                }else{
                rowHtml += '<td>' + row[col] + '</td>'; 
                }
            });
            rowHtml += '</tr>';

            if (isAnomaly) {
                anomalyHtml += rowHtml;
            } else {
                normalHtml += rowHtml;
            }
        });

        normalHtml += '</table>';
        anomalyHtml += '</table>';

        document.getElementById('normal-table').innerHTML = normalHtml;
        document.getElementById('anomaly-table').innerHTML = anomalyHtml;
}
