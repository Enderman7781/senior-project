$(document).ready(function () {
    $('#predict').click(function () {
        $(this).attr('disabled', 'true');  // 禁用按鈕
        $('#spinner').addClass('spinner-border spinner-border-sm');  // 顯示加載動畫
        $('#btn-text').text('預測中...');

        console.log('Predicting');
        getPrediction().always(() => {
            $('#spinner').removeClass('spinner-border spinner-border-sm');  // 移除加載動畫
            $(this).removeAttr('disabled');  // 重新啟用按鈕
            $('#btn-text').text('開始預測');
        });
    });
});
/*
function getCurrentDateTime() {
    const now = new Date();
    
    // 減去十分鐘 獲取最新時間
    now.setMinutes(now.getMinutes() - 10);
    const year = now.getFullYear();
    const month = (now.getMonth() + 1).toString().padStart(2, '0'); // 月份從0開始，所以需要加1
    const day = now.getDate().toString().padStart(2, '0');

    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const seconds = now.getSeconds().toString().padStart(2, '0');

    return {
        date: `${year}${month}${day}`,
        hour: hours,
        minute: minutes,
        second: seconds
    };
}

async function getSpeed() {
    try {
        let time = getCurrentDateTime();
        const proxyUrl = 'https://cors-anywhere.herokuapp.com/';
        const baseUrl = `https://tisvcloud.freeway.gov.tw/history/TDCS/M05A/${time.date}/${time.hour}/`;

        let floorMinute = parseInt(time.minute) - (parseInt(time.minute) % 5);
        floorMinute = floorMinute.toString().padStart(2, '0');

        let requestUrl = `${baseUrl}TDCS_M05A_${time.date}_${time.hour}${floorMinute}00.csv`;
        console.log('Fetching URL:', proxyUrl + requestUrl);

        const response = await fetch(proxyUrl + requestUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const csvData = await response.text();
        return csvData;

    } catch (error) {
        console.error('Failed to fetch speed data:', error);
        throw error;
    }
}*/
function getPrediction() {
    let option = {};
    let divs_id = ['high_loads', 'open_shoulder', 'exit_close'];
    for (const element of divs_id) {
        let values = [];
        let scope = "#" + element;
        let checkBtn = $("#" + element + "_btn");

        if (checkBtn.is(":checked")) {
            $(scope + ' input[type="checkbox"]:checked').each(function () {
                console.log("YES");
                values.push($(this).val());
            });
        }
        option[element] = values;

    }

    console.log(option);


    return $.ajax({
        url: 'http://10.21.44.135:5000/predict',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(option),
        dataType: 'json',
        success: function (responseData) {
            console.log(responseData);  // The response is already parsed as JSON
            updateTable(responseData);
            // Do further processing with responseData here
        },
        error: function (xhr, status, error) {
            console.error('Error fetching prediction:', error);
        }
    });
}

// Assuming responseData is an array of objects
function updateTable(responseData) {
    responseData = JSON.parse(responseData);
    responseData.forEach(data => {
        console.log(data);
        const start = data['起點路段'];
        const end = data['終點路段'];
        let speed = data['平均速度']; // Assuming your data object has a '速度' key for the speed information
        let predict_speed = data['預測結果']

        speed = Math.round(speed);
        predict_speed = Math.round(predict_speed);
        // Find the corresponding table cell(s) using jQuery
        const tdId = `#${start}-${end}`; // 這裡假設ID的格式為 起點-終點
        const tdPreId = tdId + "_PRE";

        // 查找對應的td元素
        const tdElement = $(tdId);
        const tdPreElement = $(tdPreId);

        if (tdElement) {
            // 更新td的內容
            if (predict_speed <= 20) {
                tdElement.attr('class', 'interchange table-primary');
            }
            else if (speed <= 40) {
                tdElement.attr('class', 'interchange table-danger');
            }
            else if (speed <= 60) {
                tdElement.attr('class', 'interchange table-warning');
            }
            else {
                tdElement.attr('class', 'interchange table-success');
            }
            tdElement.text(speed + "KM/hr");
        }
        if (tdPreElement) {
            if (predict_speed <= 20) {
                tdPreElement.attr('class', 'interchange table-primary');
            }
            else if (predict_speed <= 40) {
                tdPreElement.attr('class', 'interchange table-danger');
            }
            else if (predict_speed <= 60) {
                tdPreElement.attr('class', 'interchange table-warning');
            }
            else {
                tdPreElement.attr('class', 'interchange table-success');
            }
            tdPreElement.text(predict_speed + "KM/hr");
        }
    });
}