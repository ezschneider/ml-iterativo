# ML Interativo

Plataforma didática interativa onde o usuário pode enviar seus próprios dados, selecionar a coluna alvo (target) e visualizar de forma explicativa como um modelo de Machine Learning é treinado, avaliado e interpretado.

## 🎯 Objetivo

Criar uma ferramenta acessível e lúdica que permite:
- Envio de datasets personalizados (em `.csv`)
- Identificação automática do tipo de problema (classificação)
- Pipeline completo de treinamento com pré-processamento, GridSearch e avaliação
- Retorno visual de resultados: métricas, importância de features, SHAP, matriz de confusão

Tudo isso apresentado de forma clara para fins didáticos e exploratórios.

---

## 🧱 Tecnologias

### Backend
- **FastAPI**
- **Scikit-learn** (modelos e pipeline)
- **SHAP** (interpretação)
- **Matplotlib** (gráficos visuais em base64)
- **Pandas** / **Joblib**

### Frontend (SPA)
- **React + Vite + TypeScript**
- **Tailwind CSS**
- **Framer Motion** (animações suaves)
- **Axios** (requisições HTTP)

### Orquestração
- **Docker Compose**

---

## 🚀 Como executar o projeto localmente

### Pré-requisitos:
- Docker + Docker Compose
- Node.js + npm (para frontend)

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/ml-interativo.git
cd ml-interativo
```

### 2. Rode o backend com Docker
```bash
docker-compose up --build
```
Backend estará disponível em `http://localhost:8000`.

Você pode acessar a documentação Swagger em:
```
http://localhost:8000/docs
```

### 3. Rode o frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend acessível em: `http://localhost:5173`

---

## ✨ Funcionalidades atuais

- [x] Upload de dataset `.csv`
- [x] Escolha da coluna-alvo
- [x] Pipeline de ML automatizado com GridSearchCV
- [x] Retorno de:
  - Modelo vencedor e acurácia
  - Classification Report
  - Matriz de confusão (imagem base64)
  - Gráfico de importância de features
  - Gráfico SHAP (resumo visual)
- [x] SPA com animações suaves (Framer Motion)

---

## 📌 Estrutura de pastas

```
ml-interativo/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── ml/
│   │   │   ├── pipeline.py
│   │   │   └── interpretability.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/UploadForm.tsx
│   │   └── App.tsx
│   └── tailwind.config.js
├── docker-compose.yml
└── README.md
```

---

## 📌 Próximos passos
- [ ] Suporte a problemas de regressão
- [ ] Explicações interativas com texto simplificado
- [ ] Exportação dos resultados em PDF
- [ ] Histórico de execuções por sessão

---

## 📬 Contato
Desenvolvido por [Seu Nome] — 💼 Cientista de Dados e Desenvolvedor Python

Contribuições, sugestões e ideias são bem-vindas!

---

> "A melhor maneira de aprender Machine Learning é treinando modelos reais com os seus próprios dados."
