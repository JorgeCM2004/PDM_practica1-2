"""Detección DBSCAN y seguimiento de los vehículos del lídar de infraestructura."""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN


def timestamp_ns(path):
    """Los nanosegundos del nombre no están rellenados con ceros."""
    seconds, nanos = map(int, path.stem.split('_')[1:])
    return seconds * 1_000_000_000 + nanos


def load_points(path):
    points = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if points.size == 0:
        return np.empty((0, 3))
    if points.shape[1] != 3:
        raise ValueError(f'{path}: se esperaban las columnas x,y,z')
    return points[np.isfinite(points).all(axis=1)]


def detect(points, args):
    # En estos datos Z es la altura. Se excluye la banda próxima al suelo.
    ground = points[:, 2] <= args.ground_z
    objects = points[~ground]
    if len(objects):
        # Un punto por vóxel evita que la densidad cercana domine DBSCAN.
        _, indices = np.unique(np.floor(objects / args.voxel), axis=0,
                               return_index=True)
        objects = objects[indices]
    labels = (DBSCAN(eps=args.eps, min_samples=args.min_samples)
              .fit_predict(objects) if len(objects) else np.empty(0, dtype=int))
    detections = []
    for label in sorted(set(labels) - {-1}):
        cluster = objects[labels == label]
        low, high = cluster.min(axis=0), cluster.max(axis=0)
        size = high - low
        # También admite vehículos parcialmente visibles en los bordes.
        if len(cluster) < args.min_points or size[1] < .8 or size[2] < .6:
            continue
        detections.append({'center': (low[:2] + high[:2]) / 2,
                           'low': low, 'high': high, 'points': len(cluster)})
    return detections, int(ground.sum())


@dataclass
class Track:
    id: int
    center: np.ndarray
    time: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    streak: int = 1
    confirmed: bool = False

    def predict(self, time):
        return self.center + self.velocity * (time - self.time)


class Tracker:
    def __init__(self, gate=5., max_gap=1., min_hits=3):
        self.gate, self.max_gap, self.min_hits = gate, max_gap, min_hits
        self.active = []
        self.tracks = []

    def update(self, detections, time):
        self.active = [t for t in self.active if time - t.time <= self.max_gap]
        assigned = {}
        if self.active and detections:
            predicted = np.array([t.predict(time) for t in self.active])
            centers = np.array([d['center'] for d in detections])
            distances = np.linalg.norm(predicted[:, None] - centers[None], axis=2)
            # Bloquear parejas imposibles ANTES de la asignación global.
            cost = np.where(distances <= self.gate, distances, 1e9)
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if distances[row, col] > self.gate:
                    continue
                track = self.active[row]
                dt = time - track.time
                if dt > 0:
                    measured = (centers[col] - track.center) / dt
                    track.velocity = .8 * track.velocity + .2 * measured
                track.center, track.time = centers[col], time
                track.hits += 1
                track.streak += 1
                track.confirmed |= track.streak >= self.min_hits
                assigned[col] = track.id
        matched = set(assigned.values())
        for track in self.active:
            if track.id not in matched:
                track.streak = 0
        for i, detection in enumerate(detections):
            if i not in assigned:
                track = Track(len(self.tracks) + 1, detection['center'], time,
                              confirmed=self.min_hits <= 1)
                self.tracks.append(track)
                self.active.append(track)
                assigned[i] = track.id
        return assigned


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def visualize(files, frames, observations, args, clouds=None):
    from itertools import product

    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import key_press_handler
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Preparar las nubes para cambiar de frame sin volver a leer los CSV.
    if clouds is None:
        clouds = [load_points(path) for path in files]
    by_frame = {}
    for row in observations:
        by_frame.setdefault(row['frame'], []).append(row)
    # Muestreo independiente por vehículo: uno denso no elimina puntos de otro lejano.
    vehicle_clouds = [vehicle_points(points, by_frame.get(i, []), args.ground_z, args.voxel)
                      for i, points in enumerate(clouds)]
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(-60, 25), ylim=(0, 5), zlim=(0, 5),
           xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    # Misma escala métrica en los tres ejes: no deformar los vehículos.
    ax.set_box_aspect((85, 5, 5), zoom=1.6)
    ax.view_init(elev=25, azim=-60)
    ax.set_axis_off()
    fig.text(.5, .04, 'Izquierda: frame anterior | Derecha: frame siguiente | Arrastrar: girar | Barra de herramientas: zoom',
             ha='center')
    fig.subplots_adjust(left=0, right=1, bottom=.08, top=.92)
    artists = []
    last_drawn = None

    def draw(index):
        nonlocal last_drawn
        if index == last_drawn:
            return
        last_drawn = index
        # Solo sustituir los datos: conservar la cámara elegida con el ratón.
        for artist in artists:
            artist.remove()
        artists.clear()
        for row, points in zip(by_frame.get(index, []), vehicle_clouds[index]):
            color = plt.get_cmap('tab10')((row['id'] - 1) % 10)
            low = np.array([row[f'{axis}_min'] for axis in 'xyz'])
            high = np.array([row[f'{axis}_max'] for axis in 'xyz'])
            # Los pocos retornos de un vehículo que sale deben seguir siendo visibles.
            artists.append(ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                      s=5 if len(points) < 300 else 2, color=color,
                                      depthshade=False, clip_on=False))
            # Ocho vértices y doce aristas de la caja XYZ real.
            bits = np.array(list(product([0, 1], repeat=3)))
            corners = np.where(bits, high, low)
            edges = [[corners[i], corners[j]] for i in range(8) for j in range(i + 1, 8)
                     if np.count_nonzero(bits[i] != bits[j]) == 1]
            box = Line3DCollection(edges, colors=[color], linewidths=1.5, clip_on=False)
            ax.add_collection3d(box)
            artists.append(box)
            artists.append(ax.text(row['x'], row['y'], high[2] + .3,
                                   f"ID {row['id']}", color=color, fontweight='bold'))
        ax.set_title(f"Frame {index}/{len(frames) - 1} | t={frames[index]['tiempo_s']:.2f} s | "
                     f"Vehículos detectados: {frames[index]['vehiculos']}")

    current_index = 0

    def on_key(event):
        nonlocal current_index
        if event.key in ('left', 'right'):
            step = 1 if event.key == 'right' else -1
            next_index = max(0, min(current_index + step, len(frames) - 1))
            if next_index != current_index:
                current_index = next_index
                draw(current_index)
                fig.canvas.draw_idle()
        else:
            key_press_handler(event)

    # Las flechas no deben activar también el historial de cámara de Matplotlib.
    manager = fig.canvas.manager
    if manager is not None:
        fig.canvas.mpl_disconnect(manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', on_key)
    draw(0)
    plt.show()
    return fig


def display_points(points):
    """Limitar el dibujo de UN vehículo; conservar íntegros los poco densos."""
    step = max(1, int(np.ceil(len(points) / 6000)))
    return points[::step].astype(np.float32, copy=True)


def vehicle_points(points, rows, ground_z, margin):
    """Extraer los retornos actuales de cada caja, sin reutilizar puntos antiguos."""
    available = points[:, 2] > ground_z
    result = []
    for row in rows:
        low = np.array([row[f'{axis}_min'] for axis in 'xyz'])
        high = np.array([row[f'{axis}_max'] for axis in 'xyz'])
        # Las cajas vienen de la nube voxelizada: incluir sus puntos de borde originales.
        inside = ((points >= low - margin) & (points <= high + margin)).all(axis=1) & available
        result.append(display_points(points[inside]))
        available[inside] = False
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'resultados')
    parser.add_argument('--ground-z', type=float, default=.3, help='Altura máxima del suelo (m)')
    parser.add_argument('--voxel', type=float, default=.15, help='Tamaño del vóxel (m)')
    parser.add_argument('--eps', type=float, default=1.8, help='Radio DBSCAN (m)')
    parser.add_argument('--min-samples', type=int, default=8)
    parser.add_argument('--min-points', type=int, default=80, help='Mínimo de puntos por vehículo tras voxelizar')
    parser.add_argument('--gate', type=float, default=5., help='Distancia máxima de asociación (m)')
    parser.add_argument('--max-gap', type=float, default=1., help='Tiempo tolerado sin detección (s)')
    parser.add_argument('--min-hits', type=int, default=3, help='Detecciones consecutivas para confirmar un ID')
    parser.add_argument('--visualize', action='store_true', help='Explorar los frames en 3D con las flechas izquierda/derecha')
    args = parser.parse_args()
    for name in ['voxel', 'eps', 'min_samples', 'min_points', 'gate', 'max_gap', 'min_hits']:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} debe ser positivo y finito')
    if not np.isfinite(args.ground_z):
        parser.error('--ground-z debe ser finito')
    return args


def main():
    args = parse_args()
    files = sorted(args.data.glob('pointcloud_*.csv'), key=timestamp_ns)
    if not files:
        raise SystemExit(f'No se han encontrado frames en {args.data}')
    tracker = Tracker(args.gate, args.max_gap, args.min_hits)
    start = timestamp_ns(files[0])
    frames, observations = [], []
    clouds = [] if args.visualize else None
    for frame, path in enumerate(files):
        time = (timestamp_ns(path) - start) / 1e9
        points = load_points(path)
        if clouds is not None:
            clouds.append(points)
        detections, ground_count = detect(points, args)
        ids = tracker.update(detections, time)
        frames.append({'frame': frame, 'archivo': path.name, 'tiempo_s': time,
                       'puntos_validos': len(points), 'puntos_suelo': ground_count,
                       'candidatos_dbscan': len(detections)})
        for i, detection in enumerate(detections):
            low, high = detection['low'], detection['high']
            observations.append({'frame': frame, 'tiempo_s': time, 'id': ids[i],
                                 'x': detection['center'][0], 'y': detection['center'][1],
                                 'x_min': low[0], 'y_min': low[1], 'z_min': low[2],
                                 'x_max': high[0], 'y_max': high[1], 'z_max': high[2],
                                 'puntos': detection['points']})
        if (frame + 1) % 50 == 0:
            print(f'Procesados {frame + 1}/{len(files)} frames', flush=True)
    # Confirmación retrospectiva: conserva también las primeras observaciones.
    confirmed = {t.id for t in tracker.tracks if t.confirmed}
    observations = [row for row in observations if row['id'] in confirmed]
    for frame in frames:
        ids = sorted(row['id'] for row in observations if row['frame'] == frame['frame'])
        frame.update(vehiculos=len(ids), ids=';'.join(map(str, ids)))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'conteo_frames.csv', list(frames[0]), frames)
    write_csv(args.output / 'tracking.csv',
              ['frame', 'tiempo_s', 'id', 'x', 'y', 'x_min', 'y_min', 'z_min',
               'x_max', 'y_max', 'z_max', 'puntos'], observations)
    summary = (f'Frames: {len(files)}\nDuración: {frames[-1]["tiempo_s"]:.3f} s\n'
               f'Vehículos únicos (IDs confirmados): {len(confirmed)}\n'
               f'Máximo de vehículos detectados en un frame: {max(f["vehiculos"] for f in frames)}\n'
               f'Suelo: z <= {args.ground_z} m; vóxel: {args.voxel} m\n'
               f'DBSCAN: eps={args.eps} m; min_samples={args.min_samples}; '
               f'min_points={args.min_points}\n'
               f'Tracking: gate={args.gate} m; max_gap={args.max_gap} s; '
               f'min_hits={args.min_hits}\n')
    (args.output / 'resumen.txt').write_text(summary, encoding='utf-8')
    print(summary)
    if args.visualize:
        visualize(files, frames, observations, args, clouds)


if __name__ == "__main__":
    main()
