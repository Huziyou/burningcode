let debounceTimer;
let storyContext = ""; // 用于存储故事的完整上下文
let lastUserInput = ""; // 用于存储上一次用户的输入
const responseElement = document.getElementById('storyOutput');
const inputElement = document.getElementById('inputText');

// 设置定时器，每5秒自动请求一次GPT-3生成故事
setInterval(() => {
    const userInput = inputElement.value;

    // 检查用户是否有新的输入或是否清空了输入
    if (userInput === lastUserInput) {
        return; // 如果用户输入没有变化，不进行任何操作
    }

    // 如果用户清空了输入，清空故事内容
    if (!userInput.trim()) {
        storyContext = "";
        responseElement.innerText = "";
    } else {
        // 用户有新的输入，生成新的故事部分
        generateStory(userInput);
    }

    lastUserInput = userInput; // 更新上一次用户的输入
}, 8000); // 每5秒执行一次

async function generateStory(userInput) {
    if (!userInput.trim()) {
        responseElement.innerText = '请输入一些文本来开始故事。';
        return;
    }

    // 结合用户输入和现有故事上下文
    const fullPrompt = `${storyContext} ${createPrompt(userInput)}`;
    
    const data = {
        model: "gpt-3.5-turbo",
        prompt: fullPrompt,
        max_tokens: 50,
        temperature: 0.7,
    };

    try {
        const response = await fetch('https://api.chatanywhere.cn/v1/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer sk-zIeVmJwI9LenUlvMqoDyPw39bAjXAgushaxwhVYcGtEiF7Oh' // 用你的API密钥替换
            },
            body: JSON.stringify(data)
        });

        const responseData = await response.json();
        // 更新故事上下文
        storyContext += ` ${responseData.choices[0].text}`;
        responseElement.innerText = storyContext; // 显示完整故事
    } catch (error) {
        console.error('请求GPT-3 API时发生错误:', error);
        responseElement.innerText = '无法生成故事，请稍后再试。';
    }
}


function simulateStoryGeneration(userInput) {
    if (!userInput.trim()) {
        responseElement.innerText = '请输入一些文本来开始故事。';
        return;
    }

    // 模拟一个固定的测试文本作为响应
    const testResponse = "这里是GPT-3生成的故事内容...";
    responseElement.innerText = testResponse;
}

function createPrompt(userInput) {
    // 创建引导性提示
    const storyTheme = "这是一个关于爱情的故事。";
    const storyStarter = "故事开始是：";
    const prompt = `${storyTheme} ${storyStarter} ${userInput}`;
    return prompt;
}
