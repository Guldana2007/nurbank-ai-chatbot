export async function sendMessage(message) {
  const response = await fetch('http://localhost:9000/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query: message }),
  });

  if (!response.ok) {
    throw new Error('Ошибка при получении ответа от сервера');
  }

  const data = await response.json();
  return data.answer;
}
