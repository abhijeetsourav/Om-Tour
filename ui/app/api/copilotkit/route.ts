import { NextRequest } from "next/server";
import {
  CopilotRuntime,
  GroqAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
  langGraphPlatformEndpoint,
  copilotKitEndpoint,
} from "@copilotkit/runtime";
// import OpenAI from "openai";

// const openai = new OpenAI();
// const llmAdapter = new OpenAIAdapter({ openai } as any);
const llmAdapter = new GroqAdapter({ model: "llama-3.3-70b-versatile" });
// const langsmithApiKey = process.env.LANGSMITH_API_KEY as string;
const langsmithApiKey =
  process.env.LANGCHAIN_API_KEY || process.env.LANGSMITH_API_KEY;

export const POST = async (req: NextRequest) => {
  const searchParams = req.nextUrl.searchParams;

  // Allow using a deployed LangGraph agent for monitoring (LangSmith) by
  // setting either the query param `lgcDeploymentUrl` or the env var
  // `NEXT_PUBLIC_LGC_DEPLOYMENT_URL` / `LGC_DEPLOYMENT_URL`.
  const deploymentUrl =
    searchParams.get("lgcDeploymentUrl") ||
    process.env.NEXT_PUBLIC_LGC_DEPLOYMENT_URL ||
    process.env.LGC_DEPLOYMENT_URL;

  const remoteEndpoint =
    deploymentUrl && langsmithApiKey
      ? langGraphPlatformEndpoint({
          deploymentUrl,
          langsmithApiKey,
          agents: [
            {
              name: "travel",
              description:
                "This agent helps the user plan and manage their trips",
            },
          ],
        })
      : copilotKitEndpoint({
          url:
            process.env.NEXT_PUBLIC_COPILOTKIT_RUNTIME_URL ||
            "http://localhost:8000/copilotkit",
        });

  const runtime = new CopilotRuntime({
    remoteEndpoints: [remoteEndpoint],
  });

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: llmAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
