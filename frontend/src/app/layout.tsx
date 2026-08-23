import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Providers from "./providers";
import Sidebar from "../components/layout/Sidebar";
import Header from "../components/layout/Header";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "AI Risk Manager - Abuse-Ring & Fraud Sentinel",
  description: "Production-grade frontend prototype for an AI Risk Manager decision platform.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased dark`}>
      <body className="h-full bg-background text-text-primary font-sans antialiased overflow-hidden">
        <Providers>
          <div className="flex h-screen w-screen overflow-hidden bg-background">
            {/* Sidebar navigation */}
            <Sidebar />

            {/* Main content column */}
            <div className="flex-1 flex flex-col h-full overflow-hidden">
              <Header />
              <main className="flex-1 overflow-y-auto p-8 relative">
                {children}
              </main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
