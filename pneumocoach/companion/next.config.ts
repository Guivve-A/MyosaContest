import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // El proyecto vive dentro del repo del firmware, y Next remonta hasta el
  // package-lock.json del directorio de usuario si no se le fija la raiz.
  turbopack: { root: __dirname },
};

export default nextConfig;
