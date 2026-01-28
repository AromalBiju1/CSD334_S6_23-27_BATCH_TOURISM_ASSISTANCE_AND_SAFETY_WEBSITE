import React from 'react';
import svgPaths from "./svg-adhbmn3x16";

// Configuration for the 4 feature cards
const FEATURES = [
  {
    title: "Safety Zones Map",
    desc: "Visualize cities across the globe color-coded by safety levels based on crime data.",
    path: svgPaths.p1d0fd300,
    gradient: "from-[#1d89f7] to-[#00b1db]",
    iconScale: "viewBox='0 0 34 34'"
  },
  {
    title: "Safe Route Planning",
    desc: "Get routes optimized for safety, avoiding high risk areas when possible",
    path: svgPaths.p25ed8380,
    gradient: "from-[#b949ee] to-[#ed3ba6]",
    iconScale: "viewBox='0 0 26 33'"
  },
  {
    title: "Tourist Hotspots",
    desc: "Discover popular attractions with safety ratings for each location",
    path: svgPaths.p230ea580,
    gradient: "from-[#8162f3] to-[#611d99]",
    iconScale: "viewBox='0 0 25 31'"
  },
  {
    title: "Emergency Support",
    desc: "Quick access to emergency contacts and step-by-step guidance for crisis situations",
    path: svgPaths.p3e09dc80,
    gradient: "from-[#ff3735] to-[#fd6207]",
    iconScale: "viewBox='0 0 24 30'"
  }
];

export default function SafeTravelSection() {
  return (
    <div className="bg-[#020a19] text-white min-h-screen font-sans selection:bg-[#00bd84]/30">
      
      {/* --- HERO SECTION --- */}
      <section className="relative pt-24 pb-32 px-6 overflow-hidden">
        {/* Background Glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[300px] h-[300px] bg-[#01bc82] blur-[120px] opacity-30 pointer-events-none" />

        <div className="max-w-7xl mx-auto text-center relative z-10">
          {/* Safety Priority Badge */}
          <div className="inline-flex items-center gap-3 px-5 py-2 rounded-full border border-[#01c782] bg-[#011b22] mb-10">
            <svg className="w-5 h-6" viewBox="0 0 19 23" fill="none">
               <path d={svgPaths.p1d545900} stroke="#00CD86" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="text-[#00cd86] font-['Afacad'] text-lg">Your safety is our priority</span>
          </div>

          {/* Main Headline */}
          <h1 className="text-5xl md:text-8xl font-['ADLaM_Display'] tracking-tight leading-tight mb-16">
            Travel <span className="text-[#0eb891]">Safely</span> <br className="hidden md:block" />
            Across <span className="text-[#0db89e]">The Globe</span>
          </h1>

          {/* Stats Bar */}
          <div className="flex justify-center gap-12 md:gap-32 text-[#0ab8a6] text-3xl md:text-5xl font-['ADLaM_Display'] mb-20">
            <div className="flex flex-col">50+</div>
            <div className="flex flex-col">3</div>
            <div className="flex flex-col">100+</div>
          </div>
        </div>
      </section>

      {/* --- FEATURES SECTION --- */}
      <section className="bg-[#091322] py-24 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Section Header */}
          <div className="text-center mb-20">
            <h2 className="text-3xl md:text-4xl font-['ADLaM_Display'] mb-6">
              Everything You need for a <span className="text-[#00bd84]">Safe Travel</span>
            </h2>
            <p className="max-w-3xl mx-auto text-gray-400 text-lg md:text-xl font-['ABeeZee']">
              Our platform combines safety data, navigation, and emergency support to ensure you have a worry-free travel experience
            </p>
          </div>

          {/* Responsive Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {FEATURES.map((feature, idx) => (
              <div 
                key={idx} 
                className="bg-[#121e2f] p-8 rounded-[24px] hover:bg-[#1a2b42] transition-colors group"
              >
                {/* Icon Container with Gradient */}
                <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-8 shadow-lg`}>
                  <svg className="w-7 h-7" fill="none" preserveAspectRatio="xMidYMid meet" viewBox={feature.iconScale.match(/'(.*?)'/)[1]}>
                    <path 
                      d={feature.path} 
                      stroke="white" 
                      fill={idx === 1 ? "white" : "none"} // Location icon uses fill in your original
                      strokeWidth="2" 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                    />
                  </svg>
                </div>

                <h3 className="text-2xl font-['ABeeZee'] font-bold mb-4 group-hover:text-[#00bd84] transition-colors">
                  {feature.title}
                </h3>
                <p className="text-gray-400 font-['ABeeZee'] leading-relaxed">
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* --- FOOTER NOTE --- */}
      <footer className="py-20 px-6 text-center">
        <p className="max-w-xl mx-auto text-gray-400 font-['ADLaM_Display'] text-sm md:text-base leading-relaxed opacity-80">
          We analyze data from official sources to classify cities into three safety zones, helping you make informed travel decisions
        </p>
      </footer>

    </div>
  );
}