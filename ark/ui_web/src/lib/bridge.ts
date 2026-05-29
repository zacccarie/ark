// QWebChannel bridge to Python backend.
// Exposes ark.bridge object created in ark/ui/web_bridge.py.

declare global {
  interface Window {
    qt?: { webChannelTransport: any };
    QWebChannel?: any;
    arkBridge?: any;
  }
}

let _resolveReady: (b: any) => void;
const ready = new Promise<any>((res) => (_resolveReady = res));

export function getBridge(): Promise<any> {
  return ready;
}

export function initBridge(): void {
  if (!window.qt) {
    // Dev mode mock
    console.warn("[bridge] no Qt transport — using mock");
    _resolveReady({
      list_categories: async () =>
        ["Input", "Neural", "Code", "Viz", "Output Viewer", "Multimodal", "Diffusion"],
      list_nodes_in_category: async (cat: string) => ({
        Input: ["image_input", "tabular_input", "text_input"],
        Neural: ["dense", "conv2d", "transformer_block"],
      }[cat] || []),
      log: (msg: string) => console.log("[ark]", msg),
    });
    return;
  }
  const scriptSrc = "qrc:///qtwebchannel/qwebchannel.js";
  const s = document.createElement("script");
  s.src = scriptSrc;
  s.onload = () => {
    // @ts-ignore
    new window.QWebChannel(window.qt!.webChannelTransport, (channel: any) => {
      window.arkBridge = channel.objects.bridge;
      _resolveReady(channel.objects.bridge);
    });
  };
  s.onerror = () => {
    console.error("[bridge] failed to load qwebchannel.js");
  };
  document.head.appendChild(s);
}

initBridge();
