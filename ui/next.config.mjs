/** @type {import('next').NextConfig} */
const nextConfig = {
  // Increase max event listeners to prevent memory leak warnings during development
  onDemandEntries: {
    maxInactiveAge: 60 * 1000,
    pagesBufferLength: 5,
  },
};

// Increase default max listeners for development
if (process.env.NODE_ENV === "development") {
  process.setMaxListeners(15);
}

export default nextConfig;
