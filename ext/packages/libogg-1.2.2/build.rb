
Package.define("libogg-1.2.2") do |package|
	package.variant(:all) do |platform, config|
		RExec.env(config.build_flags) do
			Dir.chdir(package.src) do
				sh("make", "clean") if File.exist? "Makefile"
				
				sh("./configure", "--prefix=#{platform.prefix}", "--disable-dependency-tracking", "--enable-shared=no", "--enable-static=yes", *config.configure)
				sh("make install")
			end
		end
	end
end
