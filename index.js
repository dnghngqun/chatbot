const path = require("path");
const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const use = require("@tensorflow-models/universal-sentence-encoder");
const tf = require("@tensorflow/tfjs-node");

const app = express();
const port = 3000;
var lastMessage = null;

// Kết nối với MongoDB
mongoose
  .connect("mongodb://localhost:27017/chatbot")
  .then(() => {
    console.log("Kết nối MongoDB thành công");
  })
  .catch((err) => {
    console.error("Kết nối MongoDB thất bại", err);
  });

// Định nghĩa schema cho câu hỏi và câu trả lời
const Schema = mongoose.Schema;
const QA_Schema = new Schema({
  question: String,
  answer: String,
});
const QA_Model = mongoose.model("studies", QA_Schema); // Sử dụng tên collection 'study'

app.use(cors());
app.use(express.json());

// Phục vụ file HTML tại trang chủ
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// Tải mô hình Universal Sentence Encoder
let encoder;
async function loadModel() {
  encoder = await use.load();
}

loadModel().then(() =>
  console.log("Mô hình Universal Sentence Encoder đã được tải")
);

// Endpoint để dự đoán câu hỏi tương tự và cập nhật dữ liệu
app.post("/chat", async (req, res) => {
  const userMessage = req.body.message;
  // Phân tích cú pháp câu trả lời người dùng
  const extractAnswer = (text) => {
    const regex = /trả lời[^:]*:\s*"([^"]*)"/i;
    const match = text.match(regex);
    return match ? match[1] : null;
  };
  console.log("userMess: ", req.body.message);
  console.log("Last mess: ", lastMessage);

  const userAnswer = extractAnswer(userMessage); // Câu trả lời từ người dùng (nếu có)
  if (!userAnswer) lastMessage = userMessage;
  console.log("Question: ", userAnswer);

  // Lấy tất cả câu hỏi từ MongoDB
  console.log("Get all question.");
  const questions = await QA_Model.find().exec();

  console.log("All question: ", questions);
  // Tính toán độ tương tự giữa câu hỏi của người dùng và tất cả câu hỏi trong MongoDB
  let bestMatch = null;
  let maxSimilarity = -Infinity; // Khoảng cách Levenshtein nhỏ nhất

  for (const qa of questions) {
    const userMessageEmbedding = await encoder.embed(userMessage);
    const questionEmbedding = await encoder.embed(qa.question);

    // Tính toán độ tương tự giữa các vector nhúng
    const cosineDistance = tf.losses.cosineDistance(
      userMessageEmbedding,
      questionEmbedding,
      0
    );

    // Chuyển khoảng cách cosine thành độ tương tự
    const similarity = 1 - cosineDistance.dataSync()[0];

    // So sánh độ tương tự để tìm câu hỏi tương tự nhất
    if (similarity > maxSimilarity) {
      maxSimilarity = similarity;
      bestMatch = qa;
    }
  }

  console.log("Max simi: ", maxSimilarity);
  console.log("Best match: ", bestMatch);
  console.log("Last question: ", lastMessage);
  // Trả lời câu hỏi dựa trên câu hỏi tương tự nhất
  let botReply = "";
  if (bestMatch && maxSimilarity < 0.1 && !userAnswer) {
    // Thay đổi ngưỡng khoảng cách tùy thuộc vào yêu cầu
    botReply = bestMatch.answer;
  } else {
    botReply =
      "Xin lỗi, tôi không hiểu câu hỏi của bạn. Nếu được, hãy dạy tôi cách trả lời.";
    // Thêm câu hỏi và câu trả lời mới vào MongoDB nếu người dùng cung cấp câu trả lời đúng định dạng
    if (userAnswer) {
      const newQA = new QA_Model({
        question: lastMessage,
        answer: userAnswer,
      });
      await newQA.save();
      botReply = "Cảm ơn bạn! Tôi đã cập nhật câu hỏi và câu trả lời của bạn.";
    }
  }

  res.json({ reply: botReply });
});

app.listen(port, () => {
  console.log(`Chatbot đang chạy trên http://localhost:${port}`);
});
