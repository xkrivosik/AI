import matplotlib.pyplot as plt
import random
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

def k_means_c(coordinates, max_iterations=100, tol=1e-4):
    # Konvertuje vstupné súradnice na numpy pole pre jednoduchšiu manipuláciu
    coordinates = np.array(coordinates)
    
    # Začneme s jedným náhodne vybraným centroidom
    centroids = [random.choice(coordinates)]
    success_rate = 0  # Inicializácia úspešnosti
    iteration = 0  # Inicializácia počtu iterácií
    
    # Cyklus pokračuje, kým nedosiahne 100% úspešnosť alebo maximálny počet iterácií
    while success_rate < 100 and iteration < max_iterations:
        iteration += 1
        print(f"Iterácia {iteration} z {max_iterations}")

        # Priradenie klastrov ku každému bodu na základe najbližšieho centroidu
        assigned_clusters = [[] for _ in range(len(centroids))]
        # Vypočíta vzdialenosti medzi každým bodom a každým centroidom
        distances = np.linalg.norm(coordinates[:, np.newaxis] - centroids, axis=2)
        # Nájde indexy najbližších centroidov pre každý bod
        nearest_centroid_indices = np.argmin(distances, axis=1)

        # Priradí každý bod do príslušného klastru na základe najbližšieho centroidu
        for idx, point in enumerate(coordinates):
            assigned_clusters[nearest_centroid_indices[idx]].append(point)
        
        # Aktualizácia centroidov na základe priemerných súradníc bodov v každom klastri
        new_centroids = []
        for cluster in assigned_clusters:
            if cluster:
                # Ak klaster obsahuje body, centroid je priemerom súradníc bodov
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                # Ak je klaster prázdny, vyberie sa nový náhodný bod ako centroid
                new_centroids.append(random.choice(coordinates))

        new_centroids = np.array(new_centroids)
        
        centroids = new_centroids

        # Výpočet úspešnosti - percento klastrov, ktoré splnili požiadavky
        success_count = 0
        total_clusters = len(assigned_clusters)

        for idx, cluster in enumerate(assigned_clusters):
            if cluster:
                # Vypočíta priemernú vzdialenosť bodov v klastri od jeho centroidu
                distances = np.linalg.norm(np.array(cluster) - centroids[idx], axis=1)
                avg_distance = np.mean(distances)
                # Považujeme klaster za "úspešný", ak priemerná vzdialenosť je ≤ 500
                if avg_distance <= 500:
                    success_count += 1

        # Výpočet percentuálnej úspešnosti
        success_rate = (success_count / total_clusters) * 100
        print(f"Úspešnosť: {success_rate:.2f}%")
        
        # Ak úspešnosť nie je 100%, pridá sa nový náhodný centroid
        if success_rate < 100:
            centroids = np.vstack([centroids, random.choice(coordinates)])

    # Návrat výsledných centroidov a priradených klastrov
    return centroids, assigned_clusters

def k_means_m(coordinates, max_iterations=100, tol=1e-4):
    # Konvertuje vstupné súradnice na numpy pole pre jednoduchšiu manipuláciu
    coordinates = np.array(coordinates)

    # Začneme s jedným náhodne vybraným medoidom
    medoids = [random.choice(coordinates)]
    success_rate = 0  # Inicializácia úspešnosti
    iteration = 0  # Inicializácia počtu iterácií

    # Cyklus pokračuje, kým nedosiahne 100% úspešnosť alebo maximálny počet iterácií
    while success_rate < 100 and iteration < max_iterations:
        iteration += 1
        print(f"Iterácia {iteration}")

        # Vypočíta vzdialenosti bodov od medoidov pomocou štvorcovanej euklidovskej vzdialenosti
        distances_to_medoids = cdist(coordinates, medoids, 'sqeuclidean')

        # Priradí každý bod k najbližšiemu medoidu na základe minimálnej vzdialenosti
        nearest_medoid_indices = np.argmin(distances_to_medoids, axis=1)
        # Vytvorí zoznam klastrov, kde každý klaster obsahuje body priradené k príslušnému medoidu
        assigned_clusters = [coordinates[nearest_medoid_indices == i] for i in range(len(medoids))]

        # Aktualizácia medoidov
        new_medoids = []
        for cluster in assigned_clusters:
            if cluster.size > 0:  # Skontroluje, či klaster nie je prázdny
                # Vypočíta párové vzdialenosti medzi bodmi v klastri
                pairwise_distances = cdist(cluster, cluster, 'sqeuclidean')
                # Vypočíta celkovú vzdialenosť od každého bodu k ostatným v klastri
                total_distances = np.sum(pairwise_distances, axis=1)
                # Nový medoid je bod s najmenšou celkovou vzdialenosťou k ostatným bodom v klastri
                new_medoids.append(cluster[np.argmin(total_distances)])
            else:
                # Ak je klaster prázdny, vyberie sa nový náhodný bod ako medoid
                new_medoids.append(random.choice(coordinates))

        new_medoids = np.array(new_medoids)

        medoids = new_medoids

        # Výpočet úspešnosti - percento klastrov, ktoré spĺňajú požiadavky
        success_rate = sum(
            # Skontroluje, či priemerná vzdialenosť bodov od medoidu v každom klastri je ≤ 500
            np.mean(cdist(cluster, [medoid], 'euclidean')) <= 500 
            for cluster, medoid in zip(assigned_clusters, medoids) if cluster.size > 0
        )
        
        # Výpočet percentuálnej úspešnosti
        success_rate = (success_rate / len(assigned_clusters)) * 100
        print(f"Úspešnosť: {success_rate:.2f}%")

        # Ak úspešnosť nie je 100%, pridá sa nový náhodný medoid
        if success_rate < 100:
            medoids = np.vstack([medoids, random.choice(coordinates)])

    # Návrat výsledných medoidov a priradených klastrov
    return medoids, assigned_clusters

def divizne(coordinates, max_iterations=100):
    # Vypočíta počiatočný centroid ako priemer všetkých súradníc
    initial_centroid = np.mean(coordinates, axis=0)
    # Ukladá centrá s informáciou o osi rozdelenia (0 = x-ová os, 1 = y-ová os)
    centroids = [(initial_centroid, 0)]
    # Vytvorí počiatočný zhluk obsahujúci všetky body
    clusters = [[point for point in coordinates]]
    iteration = 0  # Inicializuje počítadlo iterácií

    # Cyklus pokračuje, kým počet centroidov nedosiahne max_iterations
    while len(centroids) <= max_iterations:
        iteration += 1
        print(f"Iterácia {iteration} z {max_iterations}")

        # Nájde index najväčšieho zhluku
        largest_cluster_index = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        largest_cluster = clusters[largest_cluster_index]
        largest_centroid, axis = centroids[largest_cluster_index]

        # Rozdelí najväčší zhluk na dve polovice na základe aktuálnej osi
        left_half = [point for point in largest_cluster if point[axis] < largest_centroid[axis]]
        right_half = [point for point in largest_cluster if point[axis] >= largest_centroid[axis]]

        # Inicializuje nové centroidy a zhluky
        new_centroids, new_clusters = [], []
        
        # Ak ľavá polovica nie je prázdna, vytvorí centroid a pridá zhluk
        if left_half:
            left_centroid = np.mean(left_half, axis=0)
            new_centroids.append((left_centroid, 1 - axis))  # Prepne os na ďalšie delenie
            new_clusters.append(left_half)
        
        # Ak pravá polovica nie je prázdna, vytvorí centroid a pridá zhluk
        if right_half:
            right_centroid = np.mean(right_half, axis=0)
            new_centroids.append((right_centroid, 1 - axis))  # Prepne os na ďalšie delenie
            new_clusters.append(right_half)
        
        # Nahradí pôvodný centroid a zhluk novými, ktoré vznikli delením
        del centroids[largest_cluster_index]
        del clusters[largest_cluster_index]
        centroids.extend(new_centroids)
        clusters.extend(new_clusters)

        # Používa KD-Tree na efektívne priraďovanie bodov k najbližším centroidom
        points = np.array(coordinates)
        centroids_coords = np.array([centroid[0] for centroid in centroids])
        tree = KDTree(centroids_coords)
        nearest_centroids = tree.query(points)[1]

        # Aktualizuje zhluky na základe priradení z KD-Tree
        clusters = [[] for _ in centroids]
        for idx, point in enumerate(points):
            clusters[nearest_centroids[idx]].append(point)

        # Prepočíta pozície centroidov ako priemer bodov v každom klastri
        final_centroids = [np.mean(cluster, axis=0) if cluster else np.zeros_like(points[0]) for cluster in clusters]

        # Vypočíta úspešnosť - podiel klastrov, kde je priemerná vzdialenosť bodov ≤ 500
        success_count = sum(
            np.mean([np.linalg.norm(np.array(point) - centroid) for point in cluster]) <= 500
            for cluster, centroid in zip(clusters, final_centroids) if cluster
        )

        # Výpočet percentuálnej úspešnosti
        success_rate = (success_count / len(final_centroids)) * 100 if final_centroids else 0
        print(f"Úspešnosť: {success_rate:.2f}%")

        # Ukončí cyklus, ak je dosiahnutá 100% úspešnosť
        if success_rate == 100:
            break

    # Návrat konečných klastrov a centroidov
    return clusters, final_centroids

def generate_unique_colors(n):
    # Inicializujeme prázdnu množinu na uchovávanie unikátnych farieb
    colors = set()
    
    # Pokračujeme v generovaní farieb, kým nebudeme mať 'n' unikátnych farieb
    while len(colors) < n:
        # Generujeme náhodnú farbu ako tuple so 3 hodnotami (R, G, B) v rozsahu [0, 1)
        color = tuple(np.random.rand(3))
        
        # Pridáme generovanú farbu do množiny 'colors'
        colors.add(color)  # Množina automaticky zabezpečuje unikátne hodnoty
    
    # Prevod množiny farieb na zoznam a jeho vrátenie
    return list(colors)

def plot_clusters(clusters, centroids, colors):
    plt.clf()  # Vyčistí aktuálnu figúru
    for idx, cluster in enumerate(clusters):
        # Zbierame x a y súradnice každého klastra naraz
        x_vals, y_vals = zip(*cluster) if cluster else ([], [])
        plt.scatter(x_vals, y_vals, color=colors[idx], s=15)

    # Vykreslenie všetkých centroidov naraz
    centroid_x, centroid_y = zip(*centroids)
    plt.scatter(centroid_x, centroid_y, color='none', edgecolor='black', s=15)

    plt.xlim(min_coord, max_coord)
    plt.ylim(min_coord, max_coord)
    plt.pause(0.01)

choice = input("Aký spôsob?(c-centroidy, m-medoidy, d-divízne):")

# Nastavenie rozsahu súradníc
min_coord, max_coord = -5000, 5000

# Generovanie prvých 20 náhodných jedinečných bodov
coordinates = set()
while len(coordinates) < 20:
    x = random.randint(min_coord, max_coord)
    y = random.randint(min_coord, max_coord)
    coordinates.add((x, y))

coordinates = list(coordinates)

# Generovanie ďalších 40000 bodov s jedinečnými súradnicami
while len(coordinates) < 40020:
    base_x, base_y = random.choice(coordinates)

    x_max_offset = 100
    x_min_offset = -100
    y_max_offset = 100
    y_min_offset = -100

    if base_x >= 4900:
        x_max_offset = 0
    if base_x <= 100:
        x_min_offset = 0
    if base_y >= 4900:
        y_max_offset = 0
    if base_y <= 100:
        y_min_offset = 0

    x_offset = random.randint(x_min_offset, x_max_offset)
    y_offset = random.randint(y_min_offset, y_max_offset)

    new_x = base_x + x_offset
    new_y = base_y + y_offset

    if (new_x, new_y) not in coordinates:
        coordinates.append((new_x, new_y))

start_time = time.time()

if choice == "c":
    # Inicializácia s jedným počiatočným centroidom
    centroids = [(random.randint(min_coord, max_coord), random.randint(min_coord, max_coord))]
    
    # Pripraví graf pre interaktívne vykreslenie
    plt.figure(figsize=(7, 7))
    
    # Iteratívne pridávanie centroidov v rámci k-means algoritmu
    final_centroids, clusters = k_means_c(coordinates)

    # Vygenerovanie farieb podľa finálneho počtu centroidov (dynamické podľa počtu klastrov)
    colors = generate_unique_colors(len(final_centroids))

    # Kreslí výsledky po každej iterácii
    plt.clf()  # Vyčistí graf pred kreslením
    for idx, cluster in enumerate(clusters):
        if cluster:  # Skontroluje, či klaster nie je prázdny
            x_vals, y_vals = zip(*cluster)  # Extrahuje x a y hodnoty pre body v klastri
            plt.scatter(x_vals, y_vals, color=colors[idx], s=15)

    # Vykreslenie centroidov
    centroid_x, centroid_y = zip(*final_centroids)
    plt.scatter(centroid_x, centroid_y, color='none', edgecolor='black', s=15)

    plt.pause(0.01)  # Pauza pre interaktívne vykreslenie

if choice == "m":
    # Start with one random medoid
    medoids = [random.choice(coordinates)]
    
    # Run K-medoids algorithm
    final_medoids, clusters = k_means_m(coordinates)

    # Generate unique colors based on the number of non-empty clusters
    non_empty_clusters = [cluster for cluster in clusters]
    colors = generate_unique_colors(len(non_empty_clusters))  # Generate colors only for non-empty clusters

    # Plotting results
    plt.figure(figsize=(7, 7))
    plt.clf()
    
    for idx, cluster in enumerate(non_empty_clusters):
        x_vals, y_vals = zip(*cluster)
        plt.scatter(x_vals, y_vals, color=colors[idx], s=15)

    # Plotting medoids
    medoid_x, medoid_y = zip(*final_medoids)
    plt.scatter(medoid_x, medoid_y, color='none', edgecolor='black', s=15)

if choice == "d":
    clusters, centroids = divizne(coordinates)

    # Generovanie farieb pre klastre
    colors = generate_unique_colors(len(clusters))

    # Vykreslenie výsledkov
    plt.figure(figsize=(7, 7))
    plt.clf()
    for idx, cluster in enumerate(clusters):
        x_vals, y_vals = zip(*cluster) if cluster else ([], [])
        plt.scatter(x_vals, y_vals, color=colors[idx], s=15)

    # Vykreslenie centroidov
    centroid_x, centroid_y = zip(*centroids)
    plt.scatter(centroid_x, centroid_y, color='none', edgecolor='black', s=15)

    plt.pause(0.01)  # Pauza pre interaktívne vykreslenie

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Algoritmus bežal {round(elapsed_time,2)} sekúnd.")

plt.xlim(min_coord, max_coord)
plt.ylim(min_coord, max_coord)
plt.show()
