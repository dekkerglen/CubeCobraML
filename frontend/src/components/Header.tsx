/** App header — brand + nav + model selector. */
import { useState } from "react";
import { Link, NavLink } from "react-router-dom";

import { ModelSelector } from "./ModelSelector";
import { cn } from "@/lib/cn";


const NAV = [
  { to: "/cubes",      label: "Cubes" },
  { to: "/metrics",    label: "Metrics" },
  { to: "/browse",     label: "Browse" },
  { to: "/playground", label: "Playground" },
];


function navLinkClass({ isActive }: { isActive: boolean }): string {
  return cn(
    "px-2.5 py-1 text-sm font-medium rounded-md transition-colors",
    isActive
      ? "text-fg-0 bg-bg-3"
      : "text-fg-2 hover:text-fg-0 hover:bg-bg-2",
  );
}


export function Header() {
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-bg-0/95 backdrop-blur">
      <div className="container flex items-center justify-between gap-4 h-14">
        <div className="flex items-center gap-3 md:gap-6 min-w-0">
          <button
            type="button"
            onClick={() => setMenuOpen((o) => !o)}
            aria-label="Toggle navigation"
            aria-expanded={menuOpen}
            className="md:hidden p-2 -ml-2 rounded-md text-fg-2 hover:text-fg-0 hover:bg-bg-2"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="size-5 fill-current">
              {menuOpen
                ? <path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
                : <path d="M3 6h18v2H3zm0 5h18v2H3zm0 5h18v2H3z" />}
            </svg>
          </button>
          <Link to="/" className="flex items-center gap-2 group shrink-0">
            <img src="/cobra.png" alt="" className="size-7 object-contain" />
            <span className="text-fg-0 font-bold tracking-tight hidden sm:inline">CubeCobraML</span>
          </Link>
          <nav className="hidden md:flex items-center gap-1">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} className={navLinkClass}>
                {n.label}
              </NavLink>
            ))}
          </nav>
        </div>
        <ModelSelector />
      </div>
      {menuOpen && (
        <nav className="md:hidden border-t border-border bg-bg-0/95 backdrop-blur">
          <div className="container py-2 flex flex-col gap-1">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                onClick={() => setMenuOpen(false)}
                className={navLinkClass}
              >
                {n.label}
              </NavLink>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}
