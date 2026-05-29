// Phase 69 — GPT walkthrough steps (didactique FR). Mirrors bbycroft/llm-viz
// sections. Each step focuses camera on a module key from gptScene.
export type Step = {
  title: string;
  // module key prefix to focus (resolved to first matching, e.g. "attn_0")
  focus: string;
  body: string;
};

export function buildWalkthrough(nLayers: number): Step[] {
  const steps: Step[] = [
    {
      title: "Vue d'ensemble",
      focus: "__all__",
      body: "Un GPT = tour de transformations. En bas les tokens entrent, en " +
        "haut sort la prédiction. Le flux monte à travers N blocs identiques. " +
        "Chaque cellule = un nombre (activation/poids). Glisse le slider pour " +
        "voyager module par module.",
    },
    {
      title: "Embedding — tokens → vecteurs",
      focus: "embed",
      body: "Chaque token (mot/caractère) devient un vecteur de dimension C " +
        "via une table apprise (token embed, gauche). On ajoute un vecteur " +
        "de position (pos embed, droite) car l'attention n'a pas de notion " +
        "d'ordre intrinsèque. Somme = entrée du 1er bloc. C'est le seul " +
        "endroit où le texte devient nombres.",
    },
    {
      title: "Layer Norm",
      focus: "ln1_0",
      body: "Avant chaque sous-couche, on normalise : moyenne 0, variance 1 " +
        "sur la dimension C, puis re-scale/shift apprenables (γ, β). " +
        "Stabilise les gradients, accélère la convergence. GPT utilise " +
        "pre-LN (norm AVANT attention/MLP) — meilleure stabilité profonde.",
    },
    {
      title: "Self-Attention — Q, K, V",
      focus: "attn_0",
      body: "Cœur du transformer. Chaque token projette 3 vecteurs : Query " +
        "(que cherche-je ?), Key (que contiens-je ?), Value (que " +
        "transmets-je ?). Multi-têtes : H projections parallèles capturent " +
        "des relations différentes. Q·Kᵀ donne la matrice d'attention T×T " +
        "(qui regarde qui). Masquée causale : un token ne voit que le passé.",
    },
    {
      title: "Softmax — distribution d'attention",
      focus: "softmax_0",
      body: "Les scores Q·Kᵀ / √A passent en softmax par ligne → poids " +
        "positifs sommant à 1. Chaque token obtient une moyenne pondérée " +
        "des Values du passé. √A évite la saturation du softmax en haute " +
        "dimension. Résultat ré-injecté dans le flux résiduel (+).",
    },
    {
      title: "MLP — feed-forward",
      focus: "mlp_0",
      body: "Après l'attention : MLP par token. Expansion C → 4C (fc up), " +
        "non-linéarité GELU, puis 4C → C (fc down). C'est ici que vit " +
        "l'essentiel des paramètres + la 'mémoire factuelle'. Traite chaque " +
        "position indépendamment (l'attention a déjà mélangé l'info).",
    },
    {
      title: "Connexion résiduelle",
      focus: "ln2_0",
      body: "Colonne centrale translucide = flux résiduel (residual stream). " +
        "Chaque sous-couche ajoute (pas remplace) sa contribution : " +
        "x = x + attn(ln(x)) puis x = x + mlp(ln(x)). Autoroute de " +
        "gradient → permet des réseaux très profonds sans vanishing.",
    },
    {
      title: "Empilement des blocs",
      focus: "__blocks__",
      body: `Ce bloc [LN→Attn→LN→MLP] se répète ${nLayers}× ici (96× pour ` +
        "GPT-3). Les premiers blocs captent syntaxe/positions, les " +
        "profonds des concepts abstraits. Même structure, poids différents.",
    },
    {
      title: "Sortie — logits + softmax",
      focus: "output",
      body: "LN final, puis projection vers le vocabulaire (V dimensions) = " +
        "logits. Softmax → probabilité de chaque token suivant. On " +
        "échantillonne (temperature, top-k) le prochain token, on le " +
        "ré-injecte en bas, et on recommence : génération autoregressive.",
    },
  ];
  return steps;
}
