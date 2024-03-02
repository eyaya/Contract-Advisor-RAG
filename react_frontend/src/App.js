import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
//import "bootstrap/dist/css/bootstrap.min.css";
//import "bootstrap/dist/js/bootstrap.min.js";
import './App.css';

import Chat from "./pages/Chat";
import Upload from "./pages/Upload";
import Home from "./pages/Home";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/home" element={<Home />} />
        <Route path="/" element={<Chat />} />
        <Route path="/upload" element={<Upload />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;