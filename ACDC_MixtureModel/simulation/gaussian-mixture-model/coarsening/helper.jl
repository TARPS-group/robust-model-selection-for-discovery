# useful helper functions

drawnow() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
latex(s) = latexstring(replace(s," ","\\,\\,"))
logsumexp(x) = (m = maximum(x); m == -Inf ? -Inf : log.(sum(exp.(x-m))) + m)

function read_json(file)
    open(file,"r") do f
        global inDict
        inDict = JSON.parse(f)
    end
    return inDict
end

function save_res(filename, N_c, mu_c, sigma_c, p_c, z_c, L_c)
	K_c = length(find(N_c[:,10000]))
	res = Dict("K"=> K_c, "Ncomps" => N_c[:,10000], "mus" => mu_c[:,10000], "sds"=> sigma_c[:,10000], "pis"=> p_c[:,10000], "z" => z_c, "liks"=>L_c)
	str_res = JSON.json(res);
	open(filename, "w") do f
       		write(f, str_res)
	end
end
