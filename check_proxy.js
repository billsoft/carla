const https = require('https');
const http = require('http');
const { exec } = require('child_process');

const PROXY_HOST = '192.168.100.182';
const PROXY_PORT = 7890;
const TEST_HOST = 'www.google.com';
const TEST_URL = `https://${TEST_HOST}`;

console.log('--- 环境变量检查 ---');
console.log('HTTP_PROXY:', process.env.HTTP_PROXY || '未设置');
console.log('HTTPS_PROXY:', process.env.HTTPS_PROXY || '未设置');
console.log('ALL_PROXY:', process.env.ALL_PROXY || '未设置');
console.log('----------------------\n');

// 1. 测试 Node.js 环境 (使用当前环境设置)
// 注意：Node.js 标准库 https 不会自动使用环境变量代理，这可以作为给用户的演示。
function testNodeRequest(label, agent) {
    return new Promise((resolve) => {
        console.log(`[Node.js] 开始测试访问 ${TEST_URL} (${label})...`);
        const options = {
            hostname: TEST_HOST,
            port: 443,
            path: '/',
            method: 'HEAD',
            timeout: 5000,
            agent: agent
        };

        const req = https.request(options, (res) => {
            console.log(`[Node.js] ${label} 响应状态码: ${res.statusCode}`);
            console.log(`[Node.js] ${label} 结果: 成功`);
            resolve(true);
        });

        req.on('error', (e) => {
            console.log(`[Node.js] ${label} 失败: ${e.message}`);
            if (label.includes('默认') && !process.env.HTTPS_PROXY) {
                 console.log(`   (提示: Node.js 原生模块通常不自动读取系统代理变量，除非使用了特定库)`);
            }
            resolve(false);
        });
        
        req.on('timeout', () => {
            req.destroy();
            console.log(`[Node.js] ${label} 超时`);
            resolve(false);
        });

        req.end();
    });
}

// 2. 测试命令行 curl (使用当前环境设置)
function testCurl() {
    return new Promise((resolve) => {
        console.log(`\n[Command Line] 开始测试 curl 访问 ${TEST_URL} (使用当前Shell环境)...`);
        // curl 会自动读取环境变量
        exec(`curl -I -s --connect-timeout 5 ${TEST_URL}`, (error, stdout, stderr) => {
            if (error) {
                console.log(`[Command Line] curl (默认) 失败: ${error.message}`);
                resolve(false);
                return;
            }
            const statusLine = stdout.split('\r\n')[0] || stdout.split('\n')[0];
            console.log(`[Command Line] curl (默认) 成功，响应头: ${statusLine}`);
            resolve(true);
        });
    });
}

// 3. 显式使用代理测试 (测试代理服务器是否可用)
function testCurlWithExplicitProxy() {
    return new Promise((resolve) => {
        const proxyUrl = `http://${PROXY_HOST}:${PROXY_PORT}`;
        console.log(`\n[Command Line] 开始测试 curl 显式代理 (${proxyUrl}) 访问 ${TEST_URL}...`);
        exec(`curl -I -s -x ${proxyUrl} --connect-timeout 5 ${TEST_URL}`, (error, stdout, stderr) => {
            if (error) {
                console.log(`[Command Line] curl (显式指定代理) 失败: ${error.message}`);
                resolve(false);
                return;
            }
            const statusLine = stdout.split('\r\n')[0] || stdout.split('\n')[0];
            console.log(`[Command Line] curl (显式指定代理) 成功，响应头: ${statusLine}`);
            resolve(true);
        });
    });
}

async function run() {
    console.log(`目标测试地址: ${TEST_URL}`);
    console.log(`目标代理地址: ${PROXY_HOST}:${PROXY_PORT}`);
    console.log('----------------------\n');

    await testNodeRequest('默认环境 (不指定代理)');
    await testCurl();
    await testCurlWithExplicitProxy();
    
    console.log('\n--- 测试结束 ---');
}

run();
