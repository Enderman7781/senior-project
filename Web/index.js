$(document).ready(function () {
    //const url = 'https://example.com/data-source';
    getPrediction().then(fetchedData => {
        // 在這裡處理抓取到的資料
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
async function getPrediction() {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const responseData = await response.text();
        console.log('Fetched Data:', responseData);  // 打印後端返回的純文字資料
        return responseData;
    } catch (error) {
        console.error('Error fetching prediction:', error);
    }
}