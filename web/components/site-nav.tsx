import Link from "next/link";
import ThemeToggle from "@/components/theme-toggle";

const GITHUB_URL = "https://github.com/Atharva-Kanherkar/labclaw";

export default function SiteNav() {
  return (
    <header className="site-nav">
      <nav className="site-nav-inner" aria-label="Primary navigation">
        <Link href="/" className="site-brand">
          LabClaw
        </Link>
        <div className="site-nav-links">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="site-nav-link"
          >
            GitHub
          </a>
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
