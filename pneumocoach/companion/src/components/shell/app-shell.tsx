"use client";

import * as React from "react";
import { Activity, GraduationCap, Radio } from "lucide-react";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tutorial } from "@/components/tutorial/tutorial";
import { useTelemetry } from "@/hooks/use-telemetry";
import { AprendizajeTab } from "@/components/aprendizaje/aprendizaje-tab";
import { DeviceBar } from "@/components/shell/device-bar";
import { ResumenTab } from "@/components/resumen/resumen-tab";
import { SensoresTab } from "@/components/sensores/sensores-tab";

const TABS = [
  { value: "resumen", label: "Resumen", icon: Activity },
  { value: "sensores", label: "Sensores", icon: Radio },
  { value: "progreso", label: "Progreso", icon: GraduationCap },
] as const;

export function AppShell() {
  const { status } = useTelemetry();
  /* El tutorial se muestra una vez y se recuerda. En una demo el operador lo
   * salta; un paciente que abre la app por primera vez lo necesita entero. */
  const [tutorial, setTutorial] = React.useState<boolean | null>(null);
  React.useEffect(() => {
    setTutorial(window.localStorage.getItem("pc-tutorial-visto") !== "1");
  }, []);
  const cerrarTutorial = React.useCallback(() => {
    window.localStorage.setItem("pc-tutorial-visto", "1");
    setTutorial(false);
  }, []);

  const [tab, setTab] = React.useState<string>("resumen");

  if (tutorial === null) return null;   // aun no se sabe: no parpadear
  if (tutorial) {
    return (
      <main className="flex min-h-dvh items-center justify-center p-4">
        <Tutorial
          conectado={status.state === "connected"}
          onCerrar={cerrarTutorial}
        />
      </main>
    );
  }

  return (
    <div className="min-h-dvh bg-background">
      {/* Encabezado adherente. En móvil el usuario está respirando y no debería
          tener que desplazarse para ver el estado del enlace. */}
      <header className="sticky top-0 z-30 border-b bg-background/85 backdrop-blur-md">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-3 px-4 py-3">
          <div className="flex items-baseline justify-between gap-3">
            <div className="flex items-baseline gap-2">
              <span className="text-base font-semibold tracking-tight">
                PneumoCoach
              </span>
              <span className="text-[0.68rem] uppercase tracking-[0.14em] text-muted-foreground">
                Rehabilitación respiratoria
              </span>
            </div>
          </div>
          <DeviceBar />
        </div>
      </header>

      <Tabs value={tab} onValueChange={setTab} className="mx-auto w-full max-w-3xl">
        {/* Navegación: barra inferior fija en móvil, pestañas normales en
            escritorio. El pulgar no llega a la parte superior de un teléfono. */}
        <TabsList className="sticky top-[calc(env(safe-area-inset-top)+7.5rem)] z-20 mx-4 hidden w-[calc(100%-2rem)] sm:grid sm:grid-cols-3">
          {TABS.map((t) => (
            <TabsTrigger key={t.value} value={t.value} className="gap-1.5">
              <t.icon className="size-3.5" data-icon="inline-start" />
              {t.label}
            </TabsTrigger>
          ))}
        </TabsList>

        <main className="px-4 pb-28 pt-4 sm:pb-10">
          <TabsContent
            value="resumen"
            className="animate-rise focus-visible:outline-none"
          >
            <ResumenTab />
          </TabsContent>
          <TabsContent
            value="sensores"
            className="animate-rise focus-visible:outline-none"
          >
            <SensoresTab />
          </TabsContent>
          <TabsContent
            value="progreso"
            className="animate-rise focus-visible:outline-none"
          >
            <AprendizajeTab />
          </TabsContent>
        </main>

        <nav
          className="fixed inset-x-0 bottom-0 z-30 border-t bg-background/92 pb-[env(safe-area-inset-bottom)] backdrop-blur-md sm:hidden"
          aria-label="Navegación principal"
        >
          <TabsList className="grid h-auto w-full grid-cols-3 rounded-none bg-transparent p-0">
            {TABS.map((t) => (
              <TabsTrigger
                key={t.value}
                value={t.value}
                className="flex-col gap-1 rounded-none border-0 py-2.5 text-[0.66rem] data-[selected]:bg-transparent data-[selected]:text-primary data-[selected]:shadow-none"
              >
                <t.icon className="size-[1.15rem]" />
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </nav>
      </Tabs>
    </div>
  );
}
