import type { Metadata, Viewport } from "next";

import { TooltipProvider } from "@/components/ui/tooltip";
import { TelemetryProvider } from "@/hooks/use-telemetry";

import "./globals.css";

export const metadata: Metadata = {
  title: "PneumoCoach",
  description:
    "Compañera del dispositivo vestible de rehabilitación respiratoria PneumoCoach.",
  applicationName: "PneumoCoach",
  // Instalable como PWA: en Chrome de Android sale "Anadir a pantalla de
  // inicio" y queda con icono propio y a pantalla completa. Es lo que hace
  // falta para la demo; un APK exigiria envolver la app y reescribir el
  // transporte, porque el WebView de Capacitor no implementa Web Bluetooth.
  manifest: "/manifest.webmanifest",
  appleWebApp: { capable: true, title: "PneumoCoach", statusBarStyle: "black-translucent" },
  icons: {
    icon: [{ url: "/icon-192.png", sizes: "192x192" },
           { url: "/icon-512.png", sizes: "512x512" }],
    apple: "/apple-touch-icon.png",
  },
};

// Fija el tema de la barra del navegador a los mismos tokens que la app, para
// que en móvil no aparezca una franja blanca sobre un encabezado oscuro.
export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#F2F1EF" },
    { media: "(prefers-color-scheme: dark)", color: "#1B1724" },
  ],
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="es" suppressHydrationWarning>
      <body className="antialiased">
        <TelemetryProvider>
          <TooltipProvider>{children}</TooltipProvider>
        </TelemetryProvider>
      </body>
    </html>
  );
}
