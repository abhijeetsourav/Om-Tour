"use client";

import dynamic from "next/dynamic";
import { TooltipProvider } from "@/components/ui/tooltip";
import { TripsProvider } from "@/lib/hooks/use-trips";
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

// Disable server-side rendering for the MapCanvas component, this
// is because Leaflet is not compatible with server-side rendering
//
// https://github.com/PaulLeCam/react-leaflet/issues/45
let MapCanvas: any;
MapCanvas = dynamic(
  () =>
    import("@/components/MapCanvas").then((module: any) => module.MapCanvas),
  {
    ssr: false,
  },
);

export default function Home() {
  const lgcDeploymentUrl =
    globalThis.window === undefined
      ? null
      : new URL(window.location.href).searchParams.get("lgcDeploymentUrl");
  return (
    <CopilotKit
      agent="travel"
      runtimeUrl="/api/copilotkit"
      publicApiKey={process.env.NEXT_PUBLIC_CPK_PUBLIC_API_KEY}
    >
      <CopilotSidebar
        defaultOpen={false}
        clickOutsideToClose={false}
        labels={{
          title: "SAR@THi",
          initial:
            "Hi! 👋 I'm Sarathi, your AI travel planner from Om Tours. I’ll create personalized itineraries based on your preferences, budget, and schedule. Add places, adjust plans in real time, or get recommendations—let’s plan your perfect trip! ✈️🌍",
        }}
      >
        <TooltipProvider>
          <TripsProvider>
            <main className="h-screen w-screen">
              <MapCanvas />
            </main>
          </TripsProvider>
        </TooltipProvider>
      </CopilotSidebar>
    </CopilotKit>
  );
}
