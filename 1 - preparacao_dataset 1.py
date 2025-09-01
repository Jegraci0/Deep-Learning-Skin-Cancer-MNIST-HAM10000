import os
import pandas as pd
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

################################################# 1 - Split Images #################################################################

#Caminhos dos ficheiros:
#O nosso dataset inicial contem 2 pastas, cada uma com 5000 ficheiros aproximadamente 
#O dataset tambem contem um ficheiro csv que contem uma base de dados com a informação sobre as imagens do dataset, nomeadamente, 
# a que classe pertence cada imagem, pelo que, foi necessario fazer o mapeamento das imagens para a pasta da respetiva classe. 
# No total haviam 7 classes diferentes.
metadata_path = 'DatasetSkin/HAM10000_metadata.csv' # caminho para o Ficheiro CSV 
image_dirs = ['DatasetSkin/HAM10000_images_part_1', 'DatasetSkin/HAM10000_images_part_2'] # caminho para as duas pastas iniciais do dataset
output_dir = 'DatasetSkin/dataset_ com_classes' # Caminho para a pasta onde vamos organizar o nosso dataset

# Ler o Csv
df = pd.read_csv(metadata_path)

# Criar a pasta "output_dir"
os.makedirs(output_dir, exist_ok=True)

# Criar dicionário de imagens (vai procurar em ambas as pastas -> HAM10000_images_part_1 e HAM10000_images_part_2)
image_lookup = {}
for folder in image_dirs:
    for fname in os.listdir(folder):
        image_lookup[fname] = os.path.join(folder, fname)

# Iterar sobre cada linha e mover a imagem para a pasta correspondente (a pasta da classe a que pertence a imagem)
for _, row in df.iterrows():
    image_id = row['image_id'] + '.jpg'
    diagnosis = row['dx']
    src_path = image_lookup.get(image_id)
    
    if src_path and os.path.exists(src_path):
        dest_folder = os.path.join(output_dir, diagnosis)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, image_id)
        shutil.copy(src_path, dest_path)  



################################################# 2 - Oversampling #################################################################


# Lista dos caminhos para as 4 pastas que queremos fazer oversampling

class_paths = [
    'dataset_ com_classes2/akiec',
    'dataset_ com_classes2/bcc',
    'dataset_ com_classes2/df',
    'dataset_ com_classes2/vasc'
]

target_count = 1000  # Numero de imagens que queremos ter em cada classe

# data_augmentation com varias transformações
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=10,
    horizontal_flip=True,
    fill_mode='reflect'   # <- evitar o "borrado" que estava a surgir nas imagens 
)


def augment_class(class_path, target_count, datagen):
    """
    Aumenta o número de imagens numa classe até atingir o target count (usando data augmentation)
    """
    # Lista todos os ficheiros
    images = os.listdir(class_path)
    count = len(images)

    # Verifica se a classe já tem o número alvo de imagens ou mais
    if count >= target_count:
        # Se sim, imprime uma mensagem e sai da função
        print(f"{class_path} tem {count} imagens.")
        return

    # Print para ver o numero de imagens
    print(f"{class_path}: {count} -> {target_count}")

    
    index = 0
    # Loop enquanto o número de imagens for menor que o alvo
    while count < target_count:
        # Seleciona o nome da imagem a ser aumentada.
        # Usa o operador de módulo (%) para ciclar pelas imagens existentes
        # Se houver poucas imagens, elas serão usadas várias vezes.
        img_name = images[index % len(images)]
        img_path = os.path.join(class_path, img_name)
        img = load_img(img_path)
        # Converte a imagem para um array
        x = img_to_array(img)
        # Redimensiona o array
        x = x.reshape((1,) + x.shape)

        # Gerar 1 imagem com data augmentation de cada vez
        for batch in datagen.flow(
            x,              # A imagem de entrada para aumentar
            batch_size=1,   # Gera apenas 1 imagem por batch
            save_to_dir=class_path,  # Diretoria onde as imagens aumentadas serão guardadas
            save_prefix='aug',       # Prefixo para os nomes dos ficheiros das imagens geradas
            save_format='jpg'        # Formato dos ficheiros das imagens geradas
        ):
            count += 1  
            break      
        index += 1

    print(f"{class_path} tem {count} imagens.\n")

# Executa para todas as classes que vamos fazer oversampling 
for path in class_paths:
    augment_class(path)


################################################# 3 - Undersampling #################################################################

undersampling_class = 'dataset_ com_classes_balanceado/nv' #caminho para a pasta da classe que vamos fazer undersampling
target_count = 1000 # Numero de imagens desejadas para a classe 

# Lista de imagens
images = os.listdir(undersampling_class)
current_count = len(images)

# Verificação para ver se o numero de imagens ainda é maior do que o target definindo 
if current_count <= target_count:
    print(f"{undersampling_class} já tem {current_count} imagens ou menos.")
else:
    print(f"{undersampling_class}: {current_count} → {target_count}")

    # Seleciona aleatoriamente as imagens que vai manter
    images_to_keep = set(random.sample(images, target_count))

    # Apaga as restantes 
    for img in images:
        if img not in images_to_keep:
            os.remove(os.path.join(undersampling_class, img))

    print(f"{undersampling_class} agora tem {target_count} imagens.")




################################################# 4 - Split Sets #################################################################


source_dir = 'dataset_ com_classes_balanceado'
dest_dir = 'dataset_balanceado_final'
splits = ['train', 'validation', 'test'] # Pastas que precisamos para o dataset 
ratios = [0.6, 0.2, 0.2] # Definimos 60% training set, 20% test set e 20% validation set

random.seed(42) # fixada a semente para obter os mesmos resultados

# Criar sets: 60/20/20
# Percorre cada classe dentro da diretoria de origem
for class_name in os.listdir(source_dir):
    src_class_path = os.path.join(source_dir, class_name)

    # Ignora se não for uma diretoria
    if not os.path.isdir(src_class_path):
        continue

    # Lista todas as imagens na classe e baralha-as aleatoriamente
    images = os.listdir(src_class_path)
    random.shuffle(images)
    n = len(images) # Número total de imagens na classe

    # Calcula o número de imagens para cada split com base nas proporções
    counts = [int(r * n) for r in ratios]
    # Ajusta a contagem do split de treino para compensar arredondamentos, garantindo que a soma é 'n'
    counts[0] += n - sum(counts)

    # Divide as imagens em sub-listas para treino, validação e teste
    split_images = {
        'train': images[:counts[0]],
        'validation': images[counts[0]:counts[0]+counts[1]],
        'test': images[counts[0]+counts[1]:]
    }

    # Copia as imagens para as respetivas pastas de destino
    for split in splits:
        dst_class_path = os.path.join(dest_dir, split, class_name)
        os.makedirs(dst_class_path, exist_ok=True) # Cria as pastas de destino se não existirem
        for img in split_images[split]:
            # Copia cada imagem para a sua nova localização
            shutil.copy2(os.path.join(src_class_path, img), os.path.join(dst_class_path, img))


# Imprime a contagem de imagens por split e por classe para verificação
for split in splits:
    print(f" {split.upper()}")
    split_path = os.path.join(dest_dir, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            n_imgs = len(os.listdir(class_path))
            print(f"   {class_name}: {n_imgs} imagens")
    print()
