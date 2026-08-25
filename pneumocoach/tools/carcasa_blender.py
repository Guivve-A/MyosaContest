"""Modelo paramétrico del wearable PneumoCoach. Se ejecuta DENTRO de Blender.

    blender --background --factory-startup --python carcasa_blender.py -- '<json de params>'

Todas las cotas salen del brief de diseño mecánico. Los tres parámetros que
dependen de medir la placa física —Wc, Hc, Tc— son entradas; el resto se deriva.
Cambiar el footprint del carrier reajusta las cinco piezas sin tocar geometría.

Sistema de coordenadas, alineado con el marco anatómico del DSP:
    +X  lateral (hacia la izquierda del sujeto)
    +Y  superior (hacia la cabeza)
    +Z  anterior (sale del pecho)
Unidades en milímetros; la escena se escala a metros al final para que los STL
salgan con la unidad correcta.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

# --------------------------------------------------------------------------
# Parámetros por defecto
# --------------------------------------------------------------------------

DEFAULTS = {
    # MEDIR CON CALIBRE. Estos son estimaciones desde fotografía.
    "Wc": 32.0,          # ancho del carrier MYOSA
    "Hc": 32.0,          # alto del carrier
    "Tc": 12.0,          # altura total con el breakout montado
    # Derivados / de diseño
    "wall": 1.6,         # espesor de pared
    "corner_r": 4.0,     # radio de esquina
    "chest_r": 125.0,    # radio de curvatura transversal del tórax
    "gland_d": 5.0,      # diámetro del pasacables
    "magnet_d": 6.2,     # alojamiento de imán (Ø6 nominal + 0.2 de holgura)
    "magnet_h": 3.2,
    "magnet_pitch": 32.0,
    "ball_d": 4.0,       # bolas del acoplamiento cinemático
    "ball_pitch": 30.0,
    "base_w": 46.0,      # base adhesiva
    "base_t": 2.5,
    "concept": "C2",     # C2 | C3 | ambos
    "render": True,
    "stl": True,
    "outdir": "",
    "res": 1400,
}


def params() -> dict:
    p = dict(DEFAULTS)
    if "--" in sys.argv:
        raw = sys.argv[sys.argv.index("--") + 1 :]
        if raw and raw[0].strip():
            p.update(json.loads(raw[0]))
    return p


# --------------------------------------------------------------------------
# Utilidades de escena
# --------------------------------------------------------------------------


def wipe():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 0.001
    bpy.context.scene.unit_settings.length_unit = "MILLIMETERS"


def rounded_box(name, w, h, d, r, segments=6):
    """Caja con esquinas redondeadas en el plano XY (biselado sobre Z)."""
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.object
    o.name = name
    o.scale = (w, h, d)
    bpy.ops.object.transform_apply(scale=True)

    if r > 0:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        # Sólo las aristas verticales: las que son paralelas a Z.
        verticales = [
            e for e in bm.edges
            if abs((e.verts[0].co - e.verts[1].co).normalized().z) > 0.99
        ]
        bmesh.ops.bevel(
            bm, geom=verticales, offset=min(r, w / 2 - 0.1, h / 2 - 0.1),
            segments=segments, profile=0.5, affect="EDGES",
        )
        bm.to_mesh(o.data)
        bm.free()
    return o


def cyl(name, d, h, loc=(0, 0, 0), rot=(0, 0, 0), verts=48):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=d / 2, depth=h, location=loc, rotation=rot, vertices=verts
    )
    o = bpy.context.object
    o.name = name
    return o


def boolean(target, cutter, op="DIFFERENCE"):
    m = target.modifiers.new(name=f"bool_{cutter.name}", type="BOOLEAN")
    m.object = cutter
    m.operation = op
    m.solver = "EXACT"
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def carve_chest(obj, chest_r, z_back):
    """Talla la cara posterior con la curvatura transversal del tórax.

    El tórax curva de izquierda a derecha, o sea alrededor de un eje vertical.
    Por eso el cilindro va con su eje en Y (superior), no en X.
    """
    c = cyl("cut_chest", chest_r * 2, 400,
            loc=(0, 0, z_back - chest_r), rot=(math.pi / 2, 0, 0), verts=128)
    return boolean(obj, c, "DIFFERENCE")


def shade_smooth(obj, angle=1.05):
    """Suaviza solo donde el ángulo entre caras es pequeño.

    En una pieza mecánica queremos las caras planas planas y los biseles
    suaves. `shade_auto_smooth` hace exactamente eso. Ojo: en Blender 4.1+ dejó
    de ser un modificador ('SMOOTH_BY_ANGLE' ya no existe como tipo) y pasó a
    ser operador respaldado por un nodo de geometría.
    """
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.shade_auto_smooth(angle=angle)
    except (AttributeError, TypeError, RuntimeError):
        for p in obj.data.polygons:
            p.use_smooth = True


# --------------------------------------------------------------------------
# Materiales
# --------------------------------------------------------------------------


def mat(name, rgba, rough=0.42, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = rgba
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def assign(obj, material):
    obj.data.materials.clear()
    obj.data.materials.append(material)


# --------------------------------------------------------------------------
# Piezas
# --------------------------------------------------------------------------


def build_capsule(p) -> list:
    """C2 · cápsula esternal: sólo el IMU, masa mínima sobre el hueso."""
    W = p["Wc"] + 5.0
    H = p["Hc"] + 5.0
    D = p["Tc"] + 3.5
    wall = p["wall"]

    shell = rounded_box("capsula", W, H, D, p["corner_r"])
    shell.location = (0, 0, D / 2)
    bpy.ops.object.transform_apply(location=True)

    # Cavidad interna: el carrier con 0.4 mm de holgura deslizante.
    cav = rounded_box("cut_cav", p["Wc"] + 0.4, p["Hc"] + 0.4, p["Tc"] + 1.2,
                      p["corner_r"] - wall)
    cav.location = (0, 0, wall + (p["Tc"] + 1.2) / 2)
    bpy.ops.object.transform_apply(location=True)
    shell = boolean(shell, cav)

    # Pasacables con alivio de tracción, saliendo hacia abajo (6 h).
    g = cyl("cut_gland", p["gland_d"], 40, loc=(0, -H / 2, wall + p["Tc"] / 2),
            rot=(math.pi / 2, 0, 0))
    shell = boolean(shell, g)

    # Ventana posterior: quita material donde no aporta rigidez y baja la masa.
    v = rounded_box("cut_win", p["Wc"] - 10, p["Hc"] - 10, 10, 2)
    v.location = (0, 0, -2)
    bpy.ops.object.transform_apply(location=True)
    shell = boolean(shell, v)

    shell = carve_chest(shell, p["chest_r"], 0.0)
    shade_smooth(shell)
    assign(shell, mat("petg_teal", (0.06, 0.30, 0.31, 1), 0.38))
    return [shell]


def build_base(p) -> list:
    """C3 · base adhesiva con acoplamiento cinemático de tres puntos."""
    B = p["base_w"]
    T = p["base_t"]

    base = rounded_box("base_adhesiva", B, B, T, 5.0)
    base.location = (0, 0, T / 2)
    bpy.ops.object.transform_apply(location=True)

    # Alojamientos de imán, abiertos por la cara anterior.
    h = p["magnet_pitch"] / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            m = cyl("cut_mag", p["magnet_d"], p["magnet_h"] * 2,
                    loc=(sx * h, sy * h, T))
            base = boolean(base, m)

    base = carve_chest(base, p["chest_r"], 0.0)
    shade_smooth(base)
    assign(base, mat("petg_sand", (0.72, 0.66, 0.56, 1), 0.55))

    piezas = [base]

    # Las tres bolas del acoplamiento. Cono, ranura en V y plano restringen
    # exactamente los 6 grados de libertad sin sobredeterminar el apoyo.
    r = p["ball_pitch"] / 2
    for i in range(3):
        a = math.radians(90 + i * 120)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=p["ball_d"] / 2, segments=32, ring_count=16,
            location=(r * math.cos(a), r * math.sin(a), T + p["ball_d"] * 0.28),
        )
        b = bpy.context.object
        b.name = f"bola_{i + 1}"
        shade_smooth(b)
        assign(b, mat(f"acero_{i}", (0.62, 0.63, 0.65, 1), 0.22, 1.0))
        piezas.append(b)

    return piezas


def build_torso(p):
    """Referencia anatómica: un casquete con la curvatura del tórax."""
    R = p["chest_r"]
    c = cyl("torso", R * 2, 190, loc=(0, 0, -R), rot=(math.pi / 2, 0, 0), verts=160)
    cut = rounded_box("cut_t", 400, 400, 400, 0)
    cut.location = (0, 0, -200 - 1.0)
    bpy.ops.object.transform_apply(location=True)
    c = boolean(c, cut)
    shade_smooth(c)
    assign(c, mat("piel", (0.80, 0.71, 0.66, 1), 0.72))
    return c


# --------------------------------------------------------------------------
# Cámara, luces, render
# --------------------------------------------------------------------------


# Objetos de ambientación: se ven en el render pero NO deben influir en el
# encuadre. El ciclorama mide veintitantas veces la pieza, así que incluirlo en
# el bounding box aleja la cámara hasta que el producto es un punto.
NO_ENCUADRAR = {"ciclorama", "torso"}


def scene_bounds():
    """Centro y radio de las piezas reales, para encuadrar sin adivinar."""
    pts = []
    for o in bpy.context.scene.objects:
        if o.type != "MESH" or o.name in NO_ENCUADRAR:
            continue
        pts += [o.matrix_world @ Vector(v) for v in o.bound_box]
    if not pts:
        return Vector((0, 0, 0)), 50.0
    mn = Vector((min(q.x for q in pts), min(q.y for q in pts), min(q.z for q in pts)))
    mx = Vector((max(q.x for q in pts), max(q.y for q in pts), max(q.z for q in pts)))
    return (mn + mx) / 2, max((mx - mn).length / 2, 5.0)


def backdrop(center, radius):
    """Ciclorama: plano que curva hacia arriba, sin costura visible."""
    z = center.z - radius * 1.05
    bpy.ops.mesh.primitive_plane_add(size=radius * 26, location=(center.x, center.y, z))
    o = bpy.context.object
    o.name = "ciclorama"
    m = o.modifiers.new("bend", "SIMPLE_DEFORM")
    m.deform_method = "BEND"
    m.angle = math.radians(-88)
    m.deform_axis = "X"
    assign(o, mat("fondo", (0.90, 0.90, 0.89, 1), 0.85))
    return o


def studio(p, vista="tres_cuartos"):
    """Cámara auto-encuadrada más iluminación de tres puntos.

    El encuadre se calcula desde el bounding box real de la escena en vez de
    fijar una distancia: si cambian Wc/Hc/Tc o se renderiza un concepto con
    más piezas, la toma sigue estando bien compuesta sin tocar nada.
    """
    center, radius = scene_bounds()
    backdrop(center, radius)

    ANGULOS = {
        "tres_cuartos": (0.72, -0.86, 0.58),
        "frontal":      (0.02, -1.25, 0.16),
        "superior":     (0.10, -0.34, 1.28),
        "posterior":    (-0.55, 0.92, 0.44),
    }
    dirn = Vector(ANGULOS.get(vista, ANGULOS["tres_cuartos"])).normalized()

    lens = 85.0
    sensor = 36.0
    fov = 2 * math.atan(sensor / (2 * lens))
    dist = (radius / math.tan(fov / 2)) * 1.55

    bpy.ops.object.camera_add(location=center + dirn * dist)
    cam = bpy.context.object
    cam.data.lens = lens
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

    def area(offset, energy, size):
        loc = center + Vector(offset) * radius
        bpy.ops.object.light_add(type="AREA", location=loc)
        l = bpy.context.object
        l.data.energy = energy * (radius ** 2) * 0.06
        l.data.size = size * radius
        l.rotation_euler = (center - l.location).to_track_quat("-Z", "Y").to_euler()
        return l

    area((2.4, -3.0, 3.6), 5200, 5.0)    # clave
    area((-3.6, -1.9, 1.4), 1700, 6.0)   # relleno
    area((-0.9, 3.2, 2.5), 2600, 4.5)    # contraluz

    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.95, 0.96, 0.96, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.45
    bpy.context.scene.world = w

    sc = bpy.context.scene

    # Cycles en CPU, no EEVEE.
    #
    # EEVEE necesita un contexto GPU real y en `--background` sobre Windows
    # falla de forma traicionera: el operador retorna sin error, se imprime la
    # ruta de salida, y no se escribe ningún archivo. Cycles en CPU no depende
    # de contexto gráfico, así que es el único camino fiable sin interfaz y en
    # CI. Cuesta segundos, no minutos, a esta resolución y con denoising.
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = p.get("samples", 64)
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6

    sc.render.resolution_x = p["res"]
    sc.render.resolution_y = int(p["res"] * 0.72)
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    # Los nombres de "look" cambian entre versiones; usar el primero que exista.
    for look in ("AgX - Medium High Contrast", "AgX - Base Contrast",
                 "AgX - Punchy", "None"):
        try:
            sc.view_settings.look = look
            break
        except TypeError:
            continue
    return cam


def reencuadrar(cam, vista):
    """Recoloca una cámara ya creada para otra vista, manteniendo el encuadre."""
    center, radius = scene_bounds()
    ANGULOS = {
        "tres_cuartos": (0.72, -0.86, 0.58),
        "frontal":      (0.02, -1.25, 0.16),
        "superior":     (0.10, -0.34, 1.28),
        "posterior":    (-0.55, 0.92, 0.44),
    }
    dirn = Vector(ANGULOS[vista]).normalized()
    fov = 2 * math.atan(36.0 / (2 * cam.data.lens))
    cam.location = center + dirn * (radius / math.tan(fov / 2)) * 1.55
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()


def render_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print(f"RENDER {path}")


def export_stl(objs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.wm.stl_export(filepath=str(path), export_selected_objects=True,
                          global_scale=1.0)
    print(f"STL {path}")


# --------------------------------------------------------------------------


def main():
    p = params()
    # Rutas SIEMPRE absolutas.
    #
    # Blender no resuelve las rutas relativas contra el CWD del proceso: una
    # ruta como "data/mockups" en render.filepath acaba escribiendo en
    # C:\data\mockups. El operador de STL sí respeta el CWD, así que con
    # rutas relativas los dos artefactos aterrizan en sitios distintos y el
    # render parece haber fallado cuando en realidad se guardó en otro disco.
    out = (Path(p["outdir"]) if p["outdir"] else Path.cwd() / "mockups").resolve()

    wipe()
    piezas = []
    if p["concept"] in ("C2", "ambos"):
        piezas += build_capsule(p)
    if p["concept"] in ("C3", "ambos"):
        base = build_base(p)
        if p["concept"] == "ambos":
            for o in base:
                o.location.x += p["Wc"] + 26
                bpy.ops.object.transform_apply(location=True)
        piezas += base

    if p["stl"] and piezas:
        export_stl(piezas, out / f"pneumocoach_{p['concept']}.stl")

    if p["render"]:
        if p.get("torso", False):
            build_torso(p)
        vistas = p.get("vistas", ["tres_cuartos"])
        cam = studio(p, vistas[0])
        for v in vistas:
            reencuadrar(cam, v)
            render_to(out / f"pneumocoach_{p['concept']}_{v}.png")

    # Resumen dimensional: lo que Jean necesita confirmar contra el brief.
    print("PARAMS " + json.dumps({k: p[k] for k in ("Wc", "Hc", "Tc", "wall", "concept")}))
    for o in piezas:
        bb = [Vector(v) for v in o.bound_box]
        dims = (max(v.x for v in bb) - min(v.x for v in bb),
                max(v.y for v in bb) - min(v.y for v in bb),
                max(v.z for v in bb) - min(v.z for v in bb))
        print(f"PIEZA {o.name} {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} mm")


if __name__ == "__main__":
    main()
