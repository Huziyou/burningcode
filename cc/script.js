let storyContext = ""; // 用于存储故事的完整上下文
const responseElement = document.getElementById('storyOutput');
const inputElement = document.getElementById('inputText');

inputElement.addEventListener('input', () => {
  // 用户正在输入，清除定时器
  clearTimeout(debounceTimer);
  // 设置新的定时器
  debounceTimer = setTimeout(() => {
    // 用户停止输入5秒后，调用API生成故事
    if (inputElement.value.trim() !== lastUserInput) {
      generateStory(inputElement.value.trim());
      lastUserInput = inputElement.value.trim(); // 更新上一次用户的输入
    }
  }, 5000);
});

async function generateStory(userInput) {
  if (!userInput) {
    // 用户清空了输入，相应地清空故事上下文和展示内容
    storyContext = "";
    responseElement.innerText = "";
    return;
  }

  // 使用用户的当前输入作为提示
  const prompt = userInput;

  const data = {
    inputs: prompt,
    parameters: {
      max_new_tokens: 150,
      temperature: 0.7,
    },
    options: {
      use_cache: false,
    }
  };

  try {
    const response = await fetch("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1", {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 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' // 替换为你的Hugging Face API密钥
      },
      method: "POST",
      body: JSON.stringify(data)
    });

    const result = await response.json();
    // 请根据Mistral模型的具体响应格式调整下面的代码
    const generatedText = result[0]?.generated_text || "";

    if (generatedText) {
      // 更新故事上下文，并显示在页面上
      storyContext = generatedText;
      responseElement.innerText = storyContext;
    }
  } catch (error) {
    console.error('请求Hugging Face API时发生错误:', error);
    responseElement.innerText = '无法生成故事，请稍后再试。';
  }
}

// 变量来跟踪上一次用户输入
let lastUserInput = "";
// 防抖定时器
let debounceTimer;
