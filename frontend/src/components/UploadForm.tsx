import { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !target) return alert("Selecione um arquivo e a coluna alvo.");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_column", target);

    try {
      setLoading(true);
      const response = await axios.post("http://localhost:8000/upload", formData);
      setResult(response.data.result);
    } catch (error: any) {
      alert("Erro ao processar o arquivo.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-2xl font-bold mb-4"
      >
        Upload de Dataset
      </motion.h1>

      <motion.form
        onSubmit={handleSubmit}
        className="space-y-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="block w-full text-sm"
        />
        <input
          type="text"
          placeholder="Coluna alvo (ex: comprou)"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="w-full p-2 border rounded"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          {loading ? "Processando..." : "Enviar"}
        </button>
      </motion.form>

      {result && (
        <motion.div
          className="mt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-xl font-semibold">Resultado</h2>
          <p className="mt-2">Modelo: {result.best_model}</p>
          <p>Acurácia: {(result.accuracy * 100).toFixed(2)}%</p>

          <div className="mt-4">
            <h3 className="font-medium">Matriz de Confusão</h3>
            <img
              src={`data:image/png;base64,${result.confusion_matrix_image}`}
              alt="Matriz de Confusão"
              className="border rounded mt-2"
            />
          </div>

          <div className="mt-4">
            <h3 className="font-medium">Importância das Features</h3>
            <img
              src={`data:image/png;base64,${result.feature_importance_image}`}
              alt="Importância das Features"
              className="border rounded mt-2"
            />
          </div>

          {result.shap_summary_image && (
            <div className="mt-4">
              <h3 className="font-medium">Resumo SHAP</h3>
              <img
                src={`data:image/png;base64,${result.shap_summary_image}`}
                alt="SHAP Summary"
                className="border rounded mt-2"
              />
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}