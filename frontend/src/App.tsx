import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";

import { useQuery } from "@tanstack/react-query";

import { CardDrawer } from "@/components/CardDrawer";
import { CommandPalette } from "@/components/CommandPalette";
import { Header } from "@/components/Header";
import { BrowsePage } from "@/pages/Browse";
import { CardDetailPage } from "@/pages/CardDetail";
import { CubeDetailPage } from "@/pages/CubeDetail";
import { CubesPage } from "@/pages/Cubes";
import { DeckDetailPage } from "@/pages/DeckDetail";
import { DraftReplayPage } from "@/pages/DraftReplay";
import { MetricsPage } from "@/pages/Metrics";
import { PlaygroundPage } from "@/pages/Playground";
import { apiGetDraft } from "@/lib/api";
import { queryClient } from "@/lib/query";


function AppShell() {
  return (
    <div className="min-h-screen flex flex-col bg-bg-1">
      <Header />
      <main className="flex-1">
        <Routes>
          <Route path="/"                element={<Navigate to="/cubes" replace />} />
          <Route path="/metrics"         element={<MetricsPage />} />
          {/* Phase V: Distribution moved into Metrics as a tab. */}
          <Route path="/distribution"    element={<Navigate to="/metrics?tab=distribution" replace />} />
          <Route path="/collapse"        element={<Navigate to="/metrics?tab=distribution" replace />} />
          <Route path="/cubes"           element={<CubesPage />} />
          <Route path="/cube/:uuid"      element={<CubeDetailPage />} />
          <Route path="/cube/:uuid/draft/:draftId" element={<DraftReplayPage />} />
          <Route path="/cube/:uuid/deck/:idx" element={<DeckDetailPage />} />
          <Route path="/card/:idx"       element={<CardDetailPage />} />
          <Route path="/browse"          element={<BrowsePage />} />
          <Route path="/playground"      element={<PlaygroundPage />} />
          {/* Phase V: legacy /drafts/:id resolves cube_uuid then redirects to
              /cube/:uuid/draft/:id so bookmarks keep working. */}
          <Route path="/drafts/:draftId" element={<RedirectDraftToCube />} />
          {/* Phase R: standalone /drafts and /decks pages deleted — their UX
              moved into the Cube workspace tabs. Redirect old URLs to /cubes. */}
          <Route path="/drafts"          element={<Navigate to="/cubes" replace />} />
          <Route path="/decks"           element={<Navigate to="/cubes" replace />} />
        </Routes>
      </main>
      <CardDrawer />
      <CommandPalette />
    </div>
  );
}


/** Legacy /drafts/:id resolver: fetch the draft, then redirect to the
 * canonical nested URL under its cube. If the draft has no cube_uuid (very
 * old data) we fall through to the standalone replay so the page still
 * works. */
function RedirectDraftToCube() {
  const { draftId = "" } = useParams<{ draftId: string }>();
  const { data, isLoading, error } = useQuery({
    queryKey: ["draft", draftId],
    queryFn: () => apiGetDraft(draftId),
    enabled: !!draftId,
  });
  if (isLoading) return <div className="container py-8 text-fg-2">Resolving draft…</div>;
  if (error) return <div className="container py-8 text-bad">Could not load draft.</div>;
  if (data?.cube_uuid) {
    return <Navigate to={`/cube/${data.cube_uuid}/draft/${draftId}`} replace />;
  }
  return <DraftReplayPage />;
}


export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppShell />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
