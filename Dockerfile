# Usa a imagem oficial do Python
FROM python:3.119

# Instala bibliotecas necessárias para OpenCV e outras dependências
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto para o contêiner
COPY . .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Define a porta correta para o Railway
EXPOSE 10000

# Comando para iniciar a aplicação
CMD ["python", "app.py"]
